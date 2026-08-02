# L-band/NISAR PU Benchmark - datasheet (auto-generated)

**Well-posedness: VERIFIED** - 10000/10000 scenes (100%) pass the noiseless-oracle recoverability certificate (clean-oracle RMSE < 0.3); max clean-oracle RMSE = 0.0347 rad. Every scene is provably recoverable by an MCF unwrapper given correct per-edge costs.

Patch: 256x256 | accepted: 10000 | rejected (clip or not-recoverable): 332 | yield: 97%


| difficulty | n | mean coh | mean k1% | mean k2% | %scenes |k|=2 | max clean-RMSE |
|---|---|---|---|---|---|---|
| smooth | 3090 | 0.51 | 0.00 | 0.000 | 0% | 0.0000 |
| dense | 2895 | 0.48 | 0.27 | 0.128 | 100% | 0.0347 |
| mixed | 4015 | 0.50 | 0.06 | 0.018 | 24% | 0.0000 |

Recoverability certificate is computed with the oracle cost (true labels) on noiseless wrapped phase; it certifies the DATA is solvable, independent of any method under test.

## Per-regime summary (two labeled sensors)

| sensor | n | mean coh | posting m | mean k1% | residues/MP (sim) | real/MP | bracket |
|---|---|---|---|---|---|---|---|
| nisar | 5940 | 0.57 | 20.0 | 0.11 | 19535 | 18100 | yes (x1.08) |
| uavsar | 4060 | 0.40 | 6.0 | 0.10 | 33948 | 28794 | yes (x1.18) |

DC1 bracket target: each regime's simulated residue density should sit at or slightly above its real sensor (bracket reality, never undershoot). nisar real ~18,100/MP; uavsar real from calibrate_sensor.py raw .cor.grd.
