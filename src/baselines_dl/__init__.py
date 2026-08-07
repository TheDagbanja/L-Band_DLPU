"""Deep-learning baselines reproduced for a fair comparison (proposal §5, Table 3).

These are plain convolutional networks -- **no** Swin/LoRA backbone and **no**
MCF solver -- reproducing the class of prior deep-learning phase unwrappers
the letter only compared against classically. Each takes the same
``(cos psi, sin psi[, grad])`` input and sample schema as the main method
(:mod:`src.dataset`) so it drops into the existing dataloaders/eval loop, but
none of them enjoy the MCF's structural residue-free guarantee (Theorem 2.1):

* :mod:`src.baselines_dl.dlpu_cnn` -- a residual U-Net regressing the
  per-pixel wrap count ``K`` (Zhou et al.-style "DLPU" CNN).
* :mod:`src.baselines_dl.phasenet2` -- a dilated-conv / ASPP multi-scale
  network *classifying* the per-pixel wrap count over ``[-k_max, k_max]``
  (PhaseNet-2.0-style; wider receptive field than the plain U-Net above).
* :mod:`src.baselines_dl.gradient_net` -- a lightweight CNN regressing
  per-edge gradient *corrections* directly, integrated by a plain
  (non-MCF) DCT least-squares solve -- the "gradient-estimation net" of
  Table 3, illustrating why a residue-free *solver* (not just a good prior)
  matters: nothing here enforces the loop-closure constraint.

Note on the ``residues_pred`` metric (:func:`src.evaluate.count_residues`):
it measures the loop-curl of ``outputs["g_x"]``/``outputs["g_y"]``. For
``dlpu_cnn``/``phasenet2``, those gradients are the *raw differences of the
network's own assembled scalar field* ``phi_hat = psi + 2*pi*round(K)`` --
and the discrete curl of any single-valued field's own gradients is exactly
zero by construction (a topological identity, independent of whether ``K``
was predicted well). So these two will always read ``residues_pred=0``, same
as the main method's ``pixel``/``pixel_cls`` heads -- that is expected, not a
bug, and it is *not* the same claim as MCF's Theorem 2.1 guarantee. A bad
``K`` prediction here still shows up as large ``rmse``/``jump_rate``
(whole-cycle errors), just not as a "residue". ``gradient_net`` is different:
its ``g_x``/``g_y`` are the *pre-integration* per-edge predictions (not
derived from its own ``phi_hat``), so an inconsistent correction field
genuinely does show nonzero ``residues_pred`` -- this is the baseline that
actually exercises the "low-RMSE-but-not-residue-free" failure mode proposal
§5 describes.
"""
