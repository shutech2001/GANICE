"""Reproduce the empirical tables, figures, and numbers used in the paper."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

METHODS = (
    ("ganite", "GANITE"),
    ("po_flow", "PO-Flow"),
    ("diff_po", "DiffPO"),
    ("infs", "INFs"),
    ("dr_learner", "DR-Learner"),
    ("codite", "CoDiTE"),
    ("dnet", "DNet"),
    ("causal_drf", "Causal-DRF"),
    ("skcd", "SKCD (adapted)"),
    ("scigan", "SCIGAN"),
    ("drnet", "DRNet"),
    ("vcnet", "VCNet"),
    ("ganice", "GANICE"),
)
BENCHMARKS = (("IHDP", "extended_w1"), ("TCGA", "extended_w1"), ("Jobs", "rct_w1"))


def write_main_results(output_root: Path) -> None:
    """Assemble the main table from the unrounded benchmark summaries."""
    summaries = {}
    for benchmark, _ in BENCHMARKS:
        source = output_root / benchmark / "metrics.csv"
        with source.open(newline="", encoding="utf-8") as handle:
            summaries[benchmark] = {row["method"]: row for row in csv.DictReader(handle)}
    rankings = {
        benchmark: sorted(
            (float(row[metric]), method) for method, row in summaries[benchmark].items() if method in dict(METHODS)
        )
        for benchmark, metric in BENCHMARKS
    }
    fields = ["method", *[column for name, _ in BENCHMARKS for column in (name.lower(), name.lower() + "_se")]]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & IHDP & TCGA & Jobs \\",
        r"\midrule",
    ]
    with (output_root / "main_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for method, label in METHODS:
            values: dict[str, str | float] = {"method": label}
            cells = [label]
            for benchmark, metric in BENCHMARKS:
                row = summaries[benchmark].get(method)
                if row is None:
                    cells.append("--")
                    continue
                value, se = float(row[metric]), float(row[metric + "_se"])
                values[benchmark.lower()] = value
                values[benchmark.lower() + "_se"] = se
                formatted = f"{value:.3f}"
                order = [key for _, key in rankings[benchmark]]
                if method == order[0]:
                    formatted = r"\mathbf{" + formatted + "}"
                elif len(order) > 1 and method == order[1]:
                    formatted = r"\underline{" + formatted + "}"
                cells.append("$" + formatted + rf"\;({se:.3f})$")
            writer.writerow(values)
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(script: str, *arguments: object) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("benchmarks", "implementation", "sensitivity"),
        default=("benchmarks", "implementation", "sensitivity"),
    )
    parser.add_argument(
        "--tables-only", action="store_true", help="Rebuild the main table from existing benchmark CSVs."
    )
    args = parser.parse_args()
    if args.repetitions < 1 or args.parallel < 1:
        parser.error("--repetitions and --parallel must be positive")
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.tables_only:
        write_main_results(output_root)
        return
    quick = ["--quick"] if args.quick else []
    common = ["--seed", args.seed, "--parallel", args.parallel, *quick]
    if "benchmarks" in args.sections:
        run(
            "run_ganice_experiment.py",
            "--benchmark",
            "all",
            "--output-dir",
            output_root,
            "--repetitions",
            args.repetitions,
            *common,
        )
        run(
            "run_additional_baselines.py",
            "--benchmark",
            "all",
            "--existing-root",
            output_root,
            "--output-root",
            output_root,
            "--repetitions",
            args.repetitions,
            *common,
        )
        write_main_results(output_root)
    if "implementation" in args.sections:
        run("run_implementation_transfer_diagnostics.py", "--output-root", output_root / "implementation", *common)
    if "sensitivity" in args.sections:
        run("run_stress_tests.py", "--output-root", output_root / "sensitivity", "--parallel", args.parallel, *quick)
    print(f"Paper outputs saved to {output_root}")


if __name__ == "__main__":
    main()
