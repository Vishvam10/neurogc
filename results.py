from pathlib import Path
from datetime import datetime
import pandas as pd
from scipy.stats import ttest_ind

# ================= CONFIG =================

BENCHMARK_ROOT = Path("benchmarks")
OUT_MD = Path("results.md")

METRICS = ["cpu", "mem", "rps", "p95", "p99", "gc_triggered"]

METRIC_LABELS = {
    "cpu": "Avg CPU (%)",
    "mem": "Avg Memory (%)",
    "rps": "Avg RPS",
    "p95": "P95 Latency (ms)",
    "p99": "P99 Latency (ms)",
    "gc_triggered": "GC Events",
}

AGGREGATION = {
    "cpu": "mean",
    "mem": "mean",
    "rps": "mean",
    "p95": "mean",
    "p99": "mean",
    "gc_triggered": "sum",   # FIXED
}

GOOD_DIRECTION = {
    "cpu": "down",
    "mem": "down",
    "p95": "down",
    "p99": "down",
    "gc_triggered": "down",
    "rps": "up",
}

# =========================================


def parse_run_time(name: str):
    return datetime.strptime(name, "%d-%m-%Y-%H-%M")


def pct_change(old, new):
    if old == 0:
        return 0.0
    return (new - old) / old * 100


def format_delta(metric, delta):
    improved = delta < 0 if GOOD_DIRECTION[metric] == "down" else delta > 0
    emoji = "🟢" if improved else "🔴"
    return f"{emoji} {delta:+.1f}%"


def welch(prev, curr):
    _, p = ttest_ind(prev, curr, equal_var=False)
    if p < 0.05:
        return "🟢 significant"
    elif p < 0.1:
        return "🟡 weak"
    return "🔴 noise"


def pick_winner(df, metric):
    best = (
        df[metric].min()
        if GOOD_DIRECTION[metric] == "down"
        else df[metric].max()
    )
    return (
        df[df[metric] == best]
        .sort_values("arch")
        .iloc[0]["arch"]
    )


# =============== ANALYSIS =================

def analyze_arch(arch_dir: Path):
    runs = []

    for run_dir in sorted(arch_dir.iterdir()):
        csv = run_dir / "benchmark.csv"
        if not csv.exists():
            continue

        df = pd.read_csv(csv)

        summary = {
            m: getattr(df[m], AGGREGATION[m])()
            for m in METRICS
        }

        runs.append({
            "arch": arch_dir.name,
            "Run": run_dir.name,
            "time": parse_run_time(run_dir.name),
            "csv": csv,
            **summary,
        })

    if len(runs) < 2:
        return None

    df_runs = (
        pd.DataFrame(runs)
        .sort_values("time")
        .reset_index(drop=True)
    )

    delta_rows = []
    sig_rows = []

    for i in range(1, len(df_runs)):
        prev = df_runs.iloc[i - 1]
        curr = df_runs.iloc[i]

        prev_df = pd.read_csv(prev["csv"])
        curr_df = pd.read_csv(curr["csv"])

        delta_row = {"Run": curr["Run"]}
        sig_row = {"Run": curr["Run"]}

        for m in METRICS:
            delta = pct_change(prev[m], curr[m])
            delta_row[METRIC_LABELS[m]] = format_delta(m, delta)
            sig_row[METRIC_LABELS[m]] = welch(prev_df[m], curr_df[m])

        delta_rows.append(delta_row)
        sig_rows.append(sig_row)

    return (
        df_runs,
        pd.DataFrame(delta_rows),
        pd.DataFrame(sig_rows),
    )


# =============== MARKDOWN =================

def md(df):
    return df.to_markdown(index=False)


def main():
    sections = [
        "## Benchmark Results\n",
        f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
    ]

    final_runs = []

    for arch_dir in sorted(BENCHMARK_ROOT.iterdir()):
        if not arch_dir.is_dir():
            continue

        result = analyze_arch(arch_dir)
        if result is None:
            continue

        df_runs, df_delta, df_sig = result
        arch = arch_dir.name

        sections.append(f"\n### Model Architecture : `{arch.title()}`\n")

        sections.append("#### Absolute Metrics\n")
        sections.append(
            md(
                df_runs[["Run"] + METRICS]
                .rename(columns=METRIC_LABELS)
                .round(3)
            )
        )

        sections.append("\n#### Run-over-Run Changes\n")
        sections.append(md(df_delta))

        sections.append("\n#### Statistical Significance (Welch t-test)\n")
        sections.append(md(df_sig))

        final_runs.append(df_runs.iloc[-1])

    final_df = pd.DataFrame(final_runs)

    sections.append("\n## Cross-Architecture Comparison (Best Run)\n")
    sections.append(
        md(
            final_df[["arch"] + METRICS]
            .rename(columns=METRIC_LABELS)
            .round(3)
        )
    )

    sections.append("\n## 🏆 Winners\n")
    for m in METRICS:
        sections.append(
            f"- **{METRIC_LABELS[m]}** → `{pick_winner(final_df, m)}`"
        )

    OUT_MD.write_text("\n".join(sections))
    print("✅ results.md generated")


if __name__ == "__main__":
    main()
