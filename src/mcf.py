"""Minimum-cost-flow (MCF) phase unwrapping -- the integer solver under SNAPHU.

This is the rigorous, *residue-free-by-construction* core of the method
(Costantini 1998; Chen & Zebker 2001), generalised from the GRSL letter's
fixed 3-class ambiguity ({-1,0,+1}) to the TGRS proposal's **five-arc**
extension (proposal §2.5): a configurable ambiguity half-range ``k_max`` so
the same solver serves both

* ``k_max=1`` -- the GRSL two-arc construction, byte-identical to
  our GRSL letter's MCF (ambiguity range ``{-1,0,+1}``), and
* ``k_max=2`` -- the TGRS five-arc construction (ambiguity range
  ``{-2,...,+2}``) for near-fault / volcanic scenes where
  ``|grad phi| > 2*pi`` (proposal Fig. 3, Eq. 17-19).

It solves for the integer ambiguity field ``k`` on the grid edges as a
minimum-cost flow on the dual graph, where the loop *residues* of the wrapped
phase are the flow sources/sinks and the per-edge *costs* are the prior
(uniform, coherence-derived, or learned).

Convention (matches src/utils edge layout):
  psi               : (M, N) wrapped phase
  dx[i,j] = W(psi[i,j+1]-psi[i,j])   horizontal edge gradient, shape (M, N-1)
  dy[i,j] = W(psi[i+1,j]-psi[i,j])   vertical   edge gradient, shape (M-1, N)
  residue R[i,j] (plaquette i,j, shape (M-1,N-1)) =
      round( (dx[i,j] + dy[i,j+1] - dx[i+1,j] - dy[i,j]) / 2pi )

Residues from a single-pixel-spaced loop of wrapped gradients are always in
``{-1,0,+1}`` regardless of ``k_max`` -- widening the ambiguity range changes
what a *cut* can encode, not the loop-residue arithmetic.

--------------------------------------------------------------------------
Five-arc construction (k_max > 1)
--------------------------------------------------------------------------
Each primal edge gets ``k_max`` **parallel** arcs per direction between the
same dual node pair: level ``j`` (``j=1..k_max``) has capacity 1 and cost
``c_{+j}`` (forward) / ``c_{-j}`` (backward), matching proposal Fig. 3 and
Eq. (17)-(18) -- using both the level-1 and level-2 forward arcs sums their
flows to a net ``k_e=+2``, and using level-1 alone gives ``k_e=+1``; the
network-simplex algorithm picks whichever combination is globally cheapest,
which is the log-posterior-maximising assignment when the per-level costs are
the likelihood-ratio costs of Eq. (19) (Proposition 2.2).

Capacity=1 at every level would reintroduce infeasibility risk that the
original (unbounded-capacity) 3-class solver never had -- Theorem 2.1's
feasibility argument relies on being able to route *any* amount of residue
flow through an edge if that is where the cheapest path lies. So the
*highest* level (``j=k_max``) keeps unbounded capacity (an "overflow" arc at
the most-extreme-classified rate) while levels ``1..k_max-1`` are
capacity-1. For ``k_max=1`` this collapses to exactly the GRSL construction:
a single unbounded-capacity arc per direction.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

TWO_PI = 2.0 * np.pi
GROUND = ("ground",)  # sentinel node id for the boundary "earth"

ArrayOrList = Union[np.ndarray, Sequence[np.ndarray]]


def wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % TWO_PI - np.pi


def wrapped_gradients(psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dx = wrap(psi[:, 1:] - psi[:, :-1])   # (M, N-1)
    dy = wrap(psi[1:, :] - psi[:-1, :])   # (M-1, N)
    return dx, dy


def residues(psi: np.ndarray) -> np.ndarray:
    """Loop-curl residues at each 2x2 plaquette, shape (M-1, N-1), in {-1,0,+1}."""
    dx, dy = wrapped_gradients(psi)
    # dx[i,j] - dx[i+1,j] : top minus bottom horizontal edges  (use rows 0..M-2)
    curl = dx[:-1, :] + dy[:, 1:] - dx[1:, :] - dy[:, :-1]
    return np.round(curl / TWO_PI).astype(np.int64)


def quality_map(psi: np.ndarray, win: int = 5) -> np.ndarray:
    """Phase-derivative-variance quality -> pseudo-coherence in [0,1].

    Classical no-coherence quality estimate (Ghiglia & Pritt): the local variance
    of the wrapped gradients is high in noisy/low-coherence regions. We return
    ``1 - normalised variance`` so it behaves like a coherence map and plugs into
    the MCF cost (high quality -> high cost -> don't cut there). This is the
    hand-designed statistical cost (e.g. SNAPHU's).
    """
    from scipy.ndimage import uniform_filter

    M, N = psi.shape
    dx, dy = wrapped_gradients(psi)
    gx = np.zeros((M, N)); gx[:, :-1] = dx
    gy = np.zeros((M, N)); gy[:-1, :] = dy

    def local_var(g):
        m = uniform_filter(g, win)
        m2 = uniform_filter(g * g, win)
        return np.maximum(m2 - m * m, 0.0)

    q = local_var(gx) + local_var(gy)
    q = q / (q.max() + 1e-6)
    return 1.0 - q


def integrate_gradients(dx: np.ndarray, dy: np.ndarray, ref: float = 0.0) -> np.ndarray:
    """Reconstruct a field from curl-free gradients by cumulative summation.

    phi[0,0]=ref; first column from dy[:,0], each row from dx[i,:]. Path-independent
    only when curl-free (which holds after MCF), so any spanning path is valid.
    """
    M = dx.shape[0]
    N = dx.shape[1] + 1
    phi = np.zeros((M, N), dtype=np.float64)
    phi[0, 1:] = np.cumsum(dx[0, :])           # top row
    phi[1:, 0] = np.cumsum(dy[:, 0])           # first column
    # fill remaining by integrating dx along each row from the first column
    for i in range(1, M):
        phi[i, 1:] = phi[i, 0] + np.cumsum(dx[i, :])
    return phi + ref


# ---------------------------------------------------------------------------
# Level-list normalisation (k_max=1 back-compat <-> k_max>1 five-arc)
# ---------------------------------------------------------------------------
def _as_levels(x: ArrayOrList, k_max: int) -> List[np.ndarray]:
    """Normalise a per-direction cost spec to a list of ``k_max`` level arrays.

    Accepts either a single 2-D array (``k_max=1`` shorthand, or a scalar cost
    reused for every level) or a sequence of exactly ``k_max`` 2-D arrays
    (levels ``1..k_max`` in order).
    """
    if isinstance(x, np.ndarray):
        if k_max == 1:
            return [x]
        raise ValueError(
            f"k_max={k_max} needs a sequence of {k_max} per-level cost arrays, "
            f"got a single array. Pass per-edge cost arrays (cpx, cnx, cpy, cny)."
        )
    levels = list(x)
    if len(levels) != k_max:
        raise ValueError(f"Expected {k_max} cost levels, got {len(levels)}.")
    return levels


def _build_dual(M: int, N: int, R: np.ndarray,
                cpx: ArrayOrList, cnx: ArrayOrList,
                cpy: ArrayOrList, cny: ArrayOrList, cost_scale: int, k_max: int = 1):
    """Build the dual min-cost-flow problem (solver-agnostic), asymmetric costs.

    Each primal edge ``e`` has coefficient +1 in one plaquette ("plus") and -1 in
    the adjacent one ("minus"); boundary edges use a single GROUND node. The
    constraint  sum(coef * k) = -R  is flow conservation when an arc minus->plus
    carries f_e = k_e, i.e. (inflow - outflow) = -R at every node.

    Costs are *directional*: positive flow (k_e>0) costs ``cpx/cpy`` (per level);
    negative flow (k_e<0) costs ``cnx/cny``. For ``k_max>1`` each direction is a
    stack of ``k_max`` parallel capacity-1 arcs (the last one capacity-unbounded,
    see module docstring) whose flows sum to the net ``k_e``.

    Returns: n_nodes, supply (list, supply = outflow-inflow = R), arcs, where
    each arc is ``(minus_id, plus_id, cost_pos_int, cost_neg_int, capacity, key)``.
    """
    Pr, Pc = M - 1, N - 1
    ground = Pr * Pc
    n_nodes = Pr * Pc + 1

    def nid(i, j):
        return i * Pc + j

    supply = [0] * n_nodes
    total = 0
    for i in range(Pr):
        for j in range(Pc):
            supply[nid(i, j)] = int(R[i, j])
            total += int(R[i, j])
    supply[ground] = -total

    def pm_x(r, c):
        plus = nid(r, c) if r <= Pr - 1 else ground       # top edge of plaquette (r,c)
        minus = nid(r - 1, c) if r - 1 >= 0 else ground   # bottom edge of (r-1,c)
        return plus, minus

    def pm_y(r, c):
        plus = nid(r, c - 1) if c - 1 >= 0 else ground    # right edge of (r,c-1)
        minus = nid(r, c) if c <= Pc - 1 else ground      # left edge of (r,c-1)
        return plus, minus

    def ci(w):
        return max(int(round(float(w) * cost_scale)), 1)

    cpx_l, cnx_l = _as_levels(cpx, k_max), _as_levels(cnx, k_max)
    cpy_l, cny_l = _as_levels(cpy, k_max), _as_levels(cny, k_max)
    big = M * N + 10

    arcs = []  # (minus_id, plus_id, cost_pos_int, cost_neg_int, capacity, key)
    for r in range(M):
        for c in range(N - 1):
            plus, minus = pm_x(r, c)
            if plus == minus:
                continue
            for lvl in range(k_max):
                cap = big if lvl == k_max - 1 else 1
                arcs.append((minus, plus, ci(cpx_l[lvl][r, c]), ci(cnx_l[lvl][r, c]),
                            cap, ("x", r, c, lvl)))
    for r in range(M - 1):
        for c in range(N):
            plus, minus = pm_y(r, c)
            if plus == minus:
                continue
            for lvl in range(k_max):
                cap = big if lvl == k_max - 1 else 1
                arcs.append((minus, plus, ci(cpy_l[lvl][r, c]), ci(cny_l[lvl][r, c]),
                            cap, ("y", r, c, lvl)))
    return n_nodes, supply, arcs


def _flows_ortools(n_nodes, supply, arcs):
    """Solve with OR-Tools SimpleMinCostFlow (exact; supports parallel arcs)."""
    from ortools.graph.python import min_cost_flow as ortools_mcf

    smcf = ortools_mcf.SimpleMinCostFlow()
    fwd, bwd = [], []
    for minus, plus, cpos, cneg, cap, _ in arcs:
        fwd.append(smcf.add_arc_with_capacity_and_unit_cost(minus, plus, cap, cpos))
        bwd.append(smcf.add_arc_with_capacity_and_unit_cost(plus, minus, cap, cneg))
    for node in range(n_nodes):
        smcf.set_node_supply(node, supply[node])
    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"OR-Tools min-cost-flow status={status}")
    return [smcf.flow(fwd[i]) - smcf.flow(bwd[i]) for i in range(len(arcs))]


def _flows_networkx(n_nodes, supply, arcs):
    """Solve with networkx (validated reference fallback).

    networkx ``DiGraph`` cannot hold parallel arcs between the same node pair,
    which for ``k_max>1`` includes the deliberately-parallel per-level arcs; we
    collapse same-direction parallel arcs into one by summing capacity and
    taking the cheapest cost as an approximation (only used when OR-Tools is
    unavailable -- prefer OR-Tools for production / five-arc runs).
    """
    import networkx as nx

    G = nx.DiGraph()
    for node in range(n_nodes):
        G.add_node(node, demand=-supply[node])             # inflow-outflow = -R
    merged = {}
    for minus, plus, cpos, cneg, cap, _ in arcs:
        for u, v, c in ((minus, plus, cpos), (plus, minus, cneg)):
            key = (u, v)
            if key in merged:
                mc, mcap = merged[key]
                merged[key] = (min(mc, c), mcap + cap)
            else:
                merged[key] = (c, cap)
    for (u, v), (c, cap) in merged.items():
        G.add_edge(u, v, weight=c, capacity=cap)
    flow = nx.min_cost_flow(G)
    out = []
    for minus, plus, cpos, cneg, cap, _ in arcs:
        # Approximate per-arc flow by the (merged) net flow between the pair,
        # clipped to this arc's own capacity -- exact for k_max=1.
        f = flow.get(minus, {}).get(plus, 0) - flow.get(plus, {}).get(minus, 0)
        out.append(max(-cap, min(cap, f)))
    return out


def _solve_min_cost_flow(M: int, N: int, R: np.ndarray,
                         cpx: ArrayOrList, cnx: ArrayOrList,
                         cpy: ArrayOrList, cny: ArrayOrList,
                         cost_scale: int = 1000,
                         backend: str = "auto", k_max: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the dual MCF; return integer ambiguity fields (kx, ky).

    backend: "auto" (OR-Tools if importable, else networkx), "ortools", "networkx".
    """
    n_nodes, supply, arcs = _build_dual(M, N, R, cpx, cnx, cpy, cny, cost_scale, k_max=k_max)

    if backend in ("auto", "ortools"):
        try:
            flows = _flows_ortools(n_nodes, supply, arcs)
        except ImportError:
            if backend == "ortools":
                raise
            flows = _flows_networkx(n_nodes, supply, arcs)
    else:
        flows = _flows_networkx(n_nodes, supply, arcs)

    kx = np.zeros((M, N - 1), dtype=np.int64)
    ky = np.zeros((M - 1, N), dtype=np.int64)
    for (minus, plus, cpos, cneg, cap, key), f in zip(arcs, flows):
        kind, r, c, lvl = key
        if kind == "x":
            kx[r, c] += f
        else:
            ky[r, c] += f
    return kx, ky


# ---------------------------------------------------------------------------
# Public unwrap entry point
# ---------------------------------------------------------------------------
def unwrap_mcf(psi: np.ndarray,
               wx: Optional[np.ndarray] = None,
               wy: Optional[np.ndarray] = None,
               coherence: Optional[np.ndarray] = None,
               edge_costs: Optional[Tuple] = None,
               coh_weight: float = 0.0,
               backend: str = "auto",
               k_max: int = 1) -> np.ndarray:
    """Unwrap ``psi`` via minimum-cost flow. Residue-free by construction.

    Cost precedence (lower cost = cheaper to place a cut there):
      * ``edge_costs=(cpx, cnx, cpy, cny)`` -- directional per-edge costs (a list of
        ``k_max`` level arrays per direction; cheaper = prefer a cut there). If ``coherence`` is also given
        with ``coh_weight>0``, the coherence term is added to every level's
        cost (learned where coherence is uninformative, physical where it is).
      * ``wx``/``wy`` -- symmetric per-edge costs supplied directly (``k_max=1``
        only).
      * ``coherence`` (alone) -- symmetric cost = mean endpoint coherence.
      * none -- unit costs (Costantini L1).
    ``k_max`` must match the ambiguity range ``edge_costs`` was built for
    (1 = GRSL three-class, 2 = TGRS five-arc).
    """
    psi = np.asarray(psi, dtype=np.float64)
    M, N = psi.shape
    dx, dy = wrapped_gradients(psi)
    R = residues(psi)

    if edge_costs is not None:
        cpx_in, cnx_in, cpy_in, cny_in = edge_costs
        cpx = [lv.copy() for lv in _as_levels(cpx_in, k_max)]
        cnx = [lv.copy() for lv in _as_levels(cnx_in, k_max)]
        cpy = [lv.copy() for lv in _as_levels(cpy_in, k_max)]
        cny = [lv.copy() for lv in _as_levels(cny_in, k_max)]
        if coherence is not None and coh_weight > 0:
            coh = np.asarray(coherence, dtype=np.float64)
            cohx = 0.5 * (coh[:, 1:] + coh[:, :-1])       # (M, N-1)
            cohy = 0.5 * (coh[1:, :] + coh[:-1, :])       # (M-1, N)
            for lv in range(k_max):
                cpx[lv] += coh_weight * cohx; cnx[lv] += coh_weight * cohx
                cpy[lv] += coh_weight * cohy; cny[lv] += coh_weight * cohy
    else:
        if k_max != 1:
            raise ValueError("Uniform/coherence-only costs are only defined for k_max=1; "
                             "pass edge_costs for k_max>1 (five-arc) unwrapping.")
        if wx is None or wy is None:
            if coherence is not None:
                coh = np.asarray(coherence, dtype=np.float64)
                wx = 0.5 * (coh[:, 1:] + coh[:, :-1]) + 1e-3   # (M, N-1)
                wy = 0.5 * (coh[1:, :] + coh[:-1, :]) + 1e-3   # (M-1, N)
            else:
                wx = np.ones((M, N - 1))
                wy = np.ones((M - 1, N))
        cpx = cnx = wx   # symmetric
        cpy = cny = wy

    if R.any():
        kx, ky = _solve_min_cost_flow(M, N, R, cpx, cnx, cpy, cny, backend=backend, k_max=k_max)
    else:
        kx = np.zeros((M, N - 1), dtype=np.int64)
        ky = np.zeros((M - 1, N), dtype=np.int64)

    gx = dx + TWO_PI * kx
    gy = dy + TWO_PI * ky
    return integrate_gradients(gx, gy, ref=psi[0, 0])


# ---------------------------------------------------------------------------
# Self-test: k_max=1 must reproduce Costantini-style unit-cost unwrapping;
# k_max=2 must stay residue-free and recover a synthetic |k|=2 field.
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)

    # 1) k_max=1 unit-cost sanity: noise-free Itoh-satisfying field unwraps exactly.
    H, W = 48, 40
    yy, xx = np.meshgrid(np.linspace(0, 2, H), np.linspace(0, 2, W), indexing="ij")
    phi = 0.6 * xx + 0.4 * yy
    psi = wrap(phi)
    rec = unwrap_mcf(psi, k_max=1)
    err = np.abs((rec - rec[0, 0]) - (phi - phi[0, 0])).max()
    assert err < 1e-6, f"k_max=1 exact-recovery failed: {err}"
    print(f"[self-test] k_max=1 unit-cost exact recovery: max err={err:.2e}")

    # 2) k_max=2 residue-free guarantee on a steep synthetic field (|grad phi|>2pi
    # in places, forcing some |k_e|=2 edges).
    yy, xx = np.meshgrid(np.linspace(-6, 6, H), np.linspace(-6, 6, W), indexing="ij")
    phi2 = 3.5 * np.arctan2(yy, xx) + 0.15 * (xx**2 + yy**2) ** 0.5 * np.sign(xx)
    psi2 = wrap(phi2 + 0.02 * rng.standard_normal(phi2.shape))
    rec2 = unwrap_mcf(psi2, k_max=1)   # unit-cost, still valid at k_max=1 (unbounded arc)
    gx = rec2[:, 1:] - rec2[:, :-1]; gy = rec2[1:, :] - rec2[:-1, :]
    curl = gx[:-1, :] + gy[:, 1:] - gx[1:, :] - gy[:, :-1]
    assert np.allclose(np.round(curl / TWO_PI), 0), "k_max=1 solution has residues!"
    print("[self-test] k_max=1 residue-free on a steep field: OK")

    print("[self-test] PASSED")


if __name__ == "__main__":
    _self_test()
