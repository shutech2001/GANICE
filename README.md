# counterfactual-GAN

Implementation of `IGAN (Interventional GAN)` for causal conditional distribution estimation under the weighted / thinned conditional Wasserstein objective described in the draft.

The main components are:

- `counterfactual_gan/finite_state.py`: exact finite-state IGAN with statewise generators/critics and a batch conditional-Wasserstein transport regularizer.
- `counterfactual_gan/continuous.py`: continuous-conditioning IGAN with three practical single-network implementations:
  - `implementation="voronoi"`: Implementation A, soft Voronoi gated mixture-of-experts generator and critic.
  - `implementation="anisotropic"`: Implementation B, single critic with outcome Lipschitz penalty and anisotropic Besov finite-difference regularization.
  - `implementation="kernel"`: Implementation C, kernelized outcome critic with learned anchors and Besov regularization on the coefficient map.
- `counterfactual_gan/igan_core.py`: shared neural modules, kernel critics, Voronoi gates, and regularization penalties.
- `counterfactual_gan/dgp.py`: finite-state and continuous synthetic causal DGPs.
- `counterfactual_gan/benchmark_dgps.py`: GANITE-style and SCIGAN-style benchmark DGPs.
- `counterfactual_gan/ganite.py`: PyTorch GANITE baseline.
- `counterfactual_gan/scigan.py`: PyTorch SCIGAN baseline.
- `scripts/run_igan_experiment.py`: end-to-end sanity experiments and baseline comparisons.

Run the experiments with:

```bash
poetry run python scripts/run_igan_experiment.py
```

The script writes figures and `metrics.json` to `outputs/igan_experiment/` by default.
