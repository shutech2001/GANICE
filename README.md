# GANICE

Materials for "**Extended Wasserstein-GAN Approach to Causal Distribution Learning: Density-Free Estimation and Minimax Optimality**".

# What is This Repository?

This repository includes an implementation of GANICE (Generative Adversarial Network for Interventional Conditional Estimation), an extended Wasserstein GAN for causal distribution learning, as described in our paper.
It also contains the experiments and standard methods implementation presented in the paper.

### Requirements and Setup

```bash
# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter
poetry env info --path
```

The experiments expect the benchmark data under `Data/`. The repository uses:

- `Data/IHDP/sim.data` for IHDP-Dist.
- `Data/Jobs/nsw_treated.txt`, `Data/Jobs/nsw_control.txt`, and `Data/Jobs/psid_controls.txt` for Jobs/LaLonde.
- `Data/TCGA/tcga.db` or the extracted TCGA `.npz` cache for TCGA-Dose.

### Executing Experiments

The main entry point is:

```bash
poetry run python scripts/run_ganice_experiment.py [options]
```

Available benchmark choices are:

```bash
--benchmark all   # run IHDP-Dist, Jobs/LaLonde, and TCGA-Dose
--benchmark ihdp  # run IHDP-Dist only
--benchmark jobs  # run Jobs/LaLonde only
--benchmark tcga  # run TCGA-Dose only
```

For the full 100-repetition experiments used in the paper:

```bash
poetry run python scripts/run_ganice_experiment.py \
  --benchmark ihdp \
  --repetitions 100 \
  --parallel 12 \
  --output-dir outputs_ganice_100/IHDP

poetry run python scripts/run_ganice_experiment.py \
  --benchmark jobs \
  --repetitions 100 \
  --parallel 12 \
  --output-dir outputs_ganice_100/Jobs

poetry run python scripts/run_ganice_experiment.py \
  --benchmark tcga \
  --repetitions 100 \
  --parallel 12 \
  --output-dir outputs_ganice_100/TCGA
```

For a quick smoke test:

```bash
poetry run python scripts/run_ganice_experiment.py \
  --benchmark all \
  --quick \
  --repetitions 1 \
  --parallel 1 \
  --output-dir outputs_smoke
```

Useful options:

```bash
--output-dir PATH
--seed INT
--repetitions INT
--parallel INT
--quick
--jobs-observed-controls {psid,psid_nsw}
--tcga-data-dir PATH
--tcga-download
--tcga-force-download
```

`--jobs-observed-controls psid` trains the Jobs/LaLonde observational sample on NSW treated units plus PSID controls. `--jobs-observed-controls psid_nsw` additionally includes NSW train-split controls.

`--tcga-download` downloads DRNet's `tcga.db` into `--tcga-data-dir` before running TCGA-Dose. Use `--tcga-force-download` to overwrite an existing database.

The script writes `metrics.json`, repetition-level JSON files, LaTeX/CSV tables, and PDF/PNG figures into the selected output directory. Reported metrics include extended Wasserstein error, CRPS, energy distance, MMD, KS, CvM, quantile errors, tail errors, calibration diagnostics, PIT histograms, and native causal metrics.

### File Description

- `scripts/run_ganice_experiment.py`: main experiment runner for IHDP-Dist, Jobs/LaLonde, and TCGA-Dose. It trains GANICE and the baseline methods, computes all paper metrics, and writes tables and figures.
- `counterfactual_gan/ganice.py`: GANICE implementation with finite-resolution, cell-normalized extended Wasserstein training and optional factual calibration terms.
- `counterfactual_gan/neural.py`: shared neural-network modules and utilities used by GANICE.
- `counterfactual_gan/metrics.py`: Wasserstein, CRPS, energy distance, MMD, CDF, quantile, tail, calibration, and PIT metric utilities.
- `counterfactual_gan/ihdp.py`: IHDP-Dist data loader and semi-synthetic stochastic potential-outcome law.
- `counterfactual_gan/jobs.py`: Jobs/LaLonde NSW/PSID loader, splitting logic, earnings transform, and helper utilities.
- `counterfactual_gan/tcga.py`: TCGA-Dose loader, TCGA preprocessing, database download/extraction helpers, and stochastic treatment/dosage response law.
- `counterfactual_gan/ganite.py`: GANITE baseline for binary-treatment benchmarks.
- `counterfactual_gan/po_flow.py`: PO-Flow baseline with scalar continuous normalizing flows.
- `counterfactual_gan/diff_po.py`: Diff-PO baseline with conditional diffusion and propensity weighting.
- `counterfactual_gan/infs.py`: INFs baseline with nuisance and target normalizing flows.
- `counterfactual_gan/dr_learner.py`: DR-Learner baseline for binary-treatment causal mean estimation.
- `counterfactual_gan/scigan.py`: SCIGAN baseline for multiple treatments with continuous dosages.
- `counterfactual_gan/vcnet.py`: VCNet baseline for continuous dosages.
- `counterfactual_gan/drnet.py`: DRNet baseline for multiple treatments with continuous dosages.
- `counterfactual_gan/utils.py`: small array and filesystem helpers.
- `counterfactual_gan/__init__.py`: package exports for the experiment modules.

## Citation

## Contact

If you have any question, please feel free to contact: tamano-shu212@g.ecc.u-tokyo.ac.jp
