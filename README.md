# GANICE

Materials for "[**Extended Wasserstein-GAN Approach to Causal Distribution Learning: Density-Free Estimation and Minimax Optimality**](https://arxiv.org/abs/2605.10206)".

# What is This Repository?

This repository includes an implementation of GANICE (Generative Adversarial Network for Interventional Conditional Estimation), an extended Wasserstein GAN for causal distribution learning, as described in our paper.
It also contains the experiments and standard methods implementation presented in the paper.

### Requirements and Setup

```bash
# clone the repository
git clone git@github.com:shutech2001/GANICE.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter
poetry env info --path
```

The experiments expect the benchmark data under `Data/`.
The repository uses:

- `Data/IHDP/sim.data` for IHDP-Dist.
- `Data/Jobs/nsw_treated.txt`, `Data/Jobs/nsw_control.txt`, and `Data/Jobs/psid_controls.txt` for Jobs/LaLonde.
- `Data/TCGA/tcga.db` or the extracted TCGA `.npz` cache for TCGA-Dose.

### Reproducing the paper

Run the main benchmarks, implementation diagnostics, and finite-state sensitivity analysis with:

```bash
poetry run python scripts/run_paper_experiments.py --parallel 12
```

The benchmark experiments use 100 repetitions.
Implementation diagnostics use 10 paired repetitions for IHDP, 10 for Jobs, and 5 for TCGA.
Sensitivity analysis uses the six factors and three optimization seeds reported in the paper.
The existing model hyperparameters, preprocessing, data splits, and evaluation settings are retained.

The benchmark and implementation runners default to `--seed 7` and use deterministic offsets for separate random streams.
The sensitivity runner defaults to `--seed 42`;
this replaces its former date-based seed.
Existing saved sensitivity results have been retained without recomputation, so rerunning that analysis with the new default will change its numerical results.

All results are saved under `outputs/`:

```text
outputs/
  main_results.csv, main_results.tex
  IHDP/             # IHDP figures, appendix table, and underlying numbers
  Jobs/             # Jobs figures, appendix table, and underlying numbers
  TCGA/             # TCGA figures, appendix table, and underlying numbers
  implementation/   # Paired shared/routed table and transfer-discrepancy numbers
  sensitivity/      # Six-factor relative-error table and underlying numbers
```

Figures are PDFs with the filenames used in the manuscript.
Tables are provided as LaTeX and CSV;
`metrics.csv` in each benchmark folder retains the unrounded means and standard errors needed for the tables, interval widths, PIT plots, and objective ablation.
Only results used in the paper are retained.

Run a subset with:

```bash
poetry run python scripts/run_paper_experiments.py --sections benchmarks --parallel 12
poetry run python scripts/run_paper_experiments.py --sections implementation --parallel 12
poetry run python scripts/run_paper_experiments.py --sections sensitivity --parallel 12
```

To rebuild the combined main table from existing benchmark CSVs without fitting:

```bash
poetry run python scripts/run_paper_experiments.py --tables-only
```

For a shorter benchmark check, use a separate temporary directory:

```bash
poetry run python scripts/run_paper_experiments.py \
  --sections benchmarks --quick --repetitions 1 --parallel 1 \
  --output-dir /tmp/ganice-paper-check
```

Individual runners remain available:

```bash
# Original benchmark methods; --benchmark also accepts ihdp, jobs, or tcga.
poetry run python scripts/run_ganice_experiment.py --benchmark all --repetitions 100 --parallel 12

# Fit the four added binary-treatment methods and merge existing benchmark CSVs.
poetry run python scripts/run_additional_baselines.py --benchmark all --repetitions 100 --parallel 12

# Appendix implementation and sensitivity tables.
poetry run python scripts/run_implementation_transfer_diagnostics.py --parallel 12
poetry run python scripts/run_stress_tests.py --parallel 12
```

The additional-baseline runner fits CoDiTE, DNet, Causal-DRF, and SKCD on IHDP and Jobs.
TCGA uses SCIGAN, VCNet, DRNet, and GANICE.
CoDiTE and SKCD sampling adapters are described in the paper;
the displayed SKCD label is `SKCD (adapted)`.
The additional runner reads the existing benchmark CSVs and preserves the original methods' summary statistics.

The main runner also accepts `--seed`, `--jobs-observed-controls {psid,psid_nsw}`, `--tcga-data-dir`, `--tcga-download`, and `--tcga-force-download`.
The default Jobs observational sample uses NSW treated units and PSID controls.
Data downloads and extracted data caches stay under `Data/`.

### Source layout

- `scripts/run_paper_experiments.py`: runs the paper experiments and assembles the main results table.
- `scripts/run_ganice_experiment.py`: original benchmark methods and paper figures/tables.
- `scripts/run_additional_baselines.py`: added binary-treatment baselines and merged results.
- `scripts/run_implementation_transfer_diagnostics.py`: paired shared/routed implementation comparison.
- `scripts/run_stress_tests.py`: finite-state GANICE sensitivity analysis for the six reported factors.
- `estimators/ganice.py` and `estimators/neural.py`: GANICE and neural-network components.
- `estimators/ihdp.py`, `estimators/jobs.py`, and `estimators/tcga.py`: benchmark data and outcome laws.
- `estimators/mechanism_stress.py`: the bounded finite-state sensitivity outcome law.
- Other modules under `estimators/`: baseline methods, metric utilities, and shared helpers.

### Code checks

```bash
poetry install --with dev
poetry run flake8 .
poetry run ruff check .
poetry run mypy .
```

The development dependencies include the SciPy type stubs used by mypy.

## Citation
```bibtex
@article{tamano2026extended,
    author={Tamano, Shu and Imaizumi, Masaaki},
    journal={arXiv preprint arXiv:2605.10206},
    title={Extended {Wasserstein-GAN} Approach to Causal Distribution Learning: Density-Free Estimation and Minimax Optimality},
    year={2026},
}
```

## Contact

If you have any question, please feel free to contact: tamano-shu212@g.ecc.u-tokyo.ac.jp
