from pathlib import Path
from datetime import datetime
import pandas as pd

# ================= CONFIG =================

BENCHMARK_ROOT = Path("benchmarks")
OUTPUT_MD = Path("results.md")

METRICS = [
    "cpu",
    "mem",
    "rps",
    "p95",
    "p99",
    "gc_triggered",
]

AGG_METRIC = {
    "cpu": "mean",
    "mem": "mean",
    "rps": "mean",
    "p95": "mean",
    "p99": "mean",
    "gc_triggered": "sum",
}

GOOD_DIRECTION = {
    "cpu": "down",
    "mem": "down",
    "p95": "down",
    "p99": "down",
    "gc_triggered": "down",
    "rps": "up",
}

METRIC_LABELS = {
    "cpu": "Avg CPU (%)",
    "mem": "Avg Memory (%)",
    "rps": "Avg RPS",
    "p95": "P95 Latency (ms)",
    "p99": "P99 Latency (ms)",
    "gc_triggered": "GC Events",
}

# =========================================


def parse_run_time(run_name: str) -> datetime:
    return datetime.strptime(run_name, "%d-%m-%Y-%H-%M")


def load_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def aggregate(df: pd.DataFrame, metric: str):
    return getattr(df[metric], AGG_METRIC[metric])()


def pct_change(old, new):
    if old == 0:
        return 0.0
    return (new - old) / old * 100


def format_delta(metric, delta):
    good = GOOD_DIRECTION[metric]
    improved = delta < 0 if good == "down" else delta > 0
    emoji = "🟢" if improved else "🔴"
    return f"{emoji} {delta:+.1f}%"


def md_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


# ================= MAIN LOGIC =================

def main():
    md_sections = []
    all_latest_runs = []

    md_sections.append("# Benchmark Results\n")
    md_sections.append(
        f"_Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}_\n"
    )

    for arch_dir in sorted(BENCHMARK_ROOT.iterdir()):
        if not arch_dir.is_dir():
            continue

        runs = []

        for run_dir in sorted(arch_dir.iterdir()):
            csv_path = run_dir / "benchmark.csv"
            if not csv_path.exists():
                continue

            df = load_csv(csv_path)

            runs.append({
                "arch": arch_dir.name,
                "run": run_dir.name,
                "time": parse_run_time(run_dir.name),
                "csv": csv_path,
                **{m: aggregate(df, m) for m in METRICS},
            })

        if not runs:
            continue

        df_runs = pd.DataFrame(runs).sort_values("time").reset_index(drop=True)

        md_sections.append(f"\n## Architecture : `{arch_dir.name}`\n")

        # ---- Absolute metrics table ----
        abs_rows = []
        for _, r in df_runs.iterrows():
            abs_rows.append(
                [r["run"]] + [f"{r[m]:.3f}" for m in METRICS]
            )

        md_sections.append("### Absolute Metrics\n")
        md_sections.append(
            md_table(
                ["Run"] + [METRIC_LABELS[m] for m in METRICS],
                abs_rows,
            )
        )

        # ---- Delta table ----
        if len(df_runs) > 1:
            delta_rows = []
            for i in range(1, len(df_runs)):
                prev = df_runs.loc[i - 1]
                curr = df_runs.loc[i]

                row = [curr["run"]]
                for m in METRICS:
                    delta = pct_change(prev[m], curr[m])
                    row.append(format_delta(m, delta))
                delta_rows.append(row)

            md_sections.append("\n### Run-Over-Run Changes\n")
            md_sections.append(
                md_table(
                    ["Run"] + [METRIC_LABELS[m] for m in METRICS],
                    delta_rows,
                )
            )

        all_latest_runs.append(df_runs.iloc[-1])

    # ---- Cross-architecture summary ----
    if all_latest_runs:
        df_final = pd.DataFrame(all_latest_runs)

        md_sections.append("\n## Cross-Architecture Comparison (Latest Runs)\n")

        cross_rows = []
        for _, r in df_final.iterrows():
            cross_rows.append(
                [r["arch"]] + [f"{r[m]:.3f}" for m in METRICS]
            )

        md_sections.append(
            md_table(
                ["Architecture"] + [METRIC_LABELS[m] for m in METRICS],
                cross_rows,
            )
        )

        md_sections.append("\n## 🏆 Winners\n")
        for m, direction in GOOD_DIRECTION.items():
            idx = (
                df_final[m].idxmin()
                if direction == "down"
                else df_final[m].idxmax()
            )
            winner = df_final.loc[idx, "arch"]
            md_sections.append(
                f"- **{METRIC_LABELS[m]}** → `{winner}`"
            )

    OUTPUT_MD.write_text("\n".join(md_sections))
    print(f"✅ Written {OUTPUT_MD}")


if __name__ == "__main__":
    main()
