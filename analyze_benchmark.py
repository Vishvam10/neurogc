import argparse
import csv
import json
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import psutil

# Try to import torch for model metadata extraction
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_config(config_path: str = "config.json") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {"default_model": "unknown"}


def get_system_info() -> dict:
    # Get OS info
    os_name = platform.system()
    os_version = platform.version()
    os_release = platform.release()

    if os_name == "Darwin":
        os_display = f"macOS {platform.mac_ver()[0]}"
    elif os_name == "Linux":
        # Try to get distro info
        try:
            import distro

            os_display = f"{distro.name()} {distro.version()}"
        except ImportError:
            os_display = f"Linux {os_release}"
    elif os_name == "Windows":
        os_display = f"Windows {platform.win32_ver()[0]}"
    else:
        os_display = f"{os_name} {os_release}"

    # Get CPU info
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
    cpu_count_logical = psutil.cpu_count()
    try:
        cpu_freq = psutil.cpu_freq()
        cpu_freq_str = f"{cpu_freq.max:.0f} MHz" if cpu_freq else "N/A"
    except Exception:
        cpu_freq_str = "N/A"

    # Get memory info
    mem = psutil.virtual_memory()
    mem_total_gb = mem.total / (1024**3)

    # Get disk info
    disk = psutil.disk_usage("/")
    disk_total_gb = disk.total / (1024**3)

    return {
        "os": os_display,
        "os_name": os_name,
        "os_version": os_version,
        "cpu_cores": cpu_count,
        "cpu_cores_logical": cpu_count_logical,
        "cpu_freq": cpu_freq_str,
        "cpu_model": platform.processor() or "N/A",
        "memory_gb": round(mem_total_gb, 1),
        "disk_gb": round(disk_total_gb, 1),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
    }


def get_model_metadata(
    model_name: str, model_path: str = "gc_model.pth"
) -> dict:
    metadata = {
        "name": model_name,
        "version": "unknown",
        "description": "",
        "input_features": [
            "cpu",
            "mem",
            "disk_read",
            "disk_write",
            "net_sent",
            "net_recv",
        ],
        "sequence_length": 1,
    }

    # Try to load metadata from model file
    if HAS_TORCH and os.path.exists(model_path):
        try:
            checkpoint = torch.load(
                model_path, map_location="cpu", weights_only=False
            )
            if "metadata" in checkpoint:
                model_meta = checkpoint["metadata"]
                metadata.update(
                    {
                        "name": model_meta.get("name", model_name),
                        "version": model_meta.get("version", "unknown"),
                        "description": model_meta.get("description", ""),
                        "input_features": model_meta.get(
                            "input_features", metadata["input_features"]
                        ),
                        "sequence_length": model_meta.get("sequence_length", 1),
                    }
                )
        except Exception:
            pass

    # Set default descriptions based on model name
    if not metadata["description"]:
        descriptions = {
            "lstm": "LSTM-based GC predictor for temporal pattern recognition",
            "transformer": "Transformer-based GC predictor with self-attention",
            "feedforward": "Feedforward neural network GC predictor",
            "classical": "Classical ML (Random Forest) GC predictor",
        }
        metadata["description"] = descriptions.get(
            model_name.lower(), f"{model_name} GC predictor"
        )

    return metadata


def detect_model_from_checkpoint(
    model_path: str = "gc_model.pth",
) -> Optional[str]:
    if not HAS_TORCH or not os.path.exists(model_path):
        return None

    try:
        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=False
        )
        if "metadata" in checkpoint:
            return checkpoint["metadata"].get("name")
    except Exception:
        pass

    return None


def load_benchmark_csv(csv_path: str) -> list[dict]:
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            parsed_row = {
                "timestamp": float(row["timestamp"]),
                "server": row["server"],
                "cpu": float(row["cpu"]),
                "mem": float(row["mem"]),
                "disk_read": float(row["disk_read"]),
                "disk_write": float(row["disk_write"]),
                "net_sent": float(row["net_sent"]),
                "net_recv": float(row["net_recv"]),
                "rps": float(row["rps"]),
                "p95": float(row["p95"]),
                "p99": float(row["p99"]),
                "gc_triggered": row["gc_triggered"].lower() == "true",
            }
            data.append(parsed_row)
    return data


def split_by_server(data: list[dict]) -> tuple[list[dict], list[dict]]:
    with_gc = [d for d in data if d["server"] == "with_gc"]
    without_gc = [d for d in data if d["server"] == "without_gc"]
    return with_gc, without_gc


def calculate_stats(data: list[dict]) -> dict:
    if not data:
        return {}

    def avg(key):
        values = [d[key] for d in data]
        return sum(values) / len(values) if values else 0

    def percentile(key, p):
        values = sorted([d[key] for d in data])
        if not values:
            return 0
        idx = int(len(values) * p / 100)
        return values[min(idx, len(values) - 1)]

    gc_count = sum(1 for d in data if d["gc_triggered"])

    return {
        "avg_cpu": avg("cpu"),
        "avg_mem": avg("mem"),
        "avg_disk_read": avg("disk_read"),
        "avg_disk_write": avg("disk_write"),
        "avg_net_sent": avg("net_sent"),
        "avg_net_recv": avg("net_recv"),
        "avg_rps": avg("rps"),
        "avg_p95": avg("p95"),
        "avg_p99": avg("p99"),
        "p95_latency": percentile("p95", 95),
        "p99_latency": percentile("p99", 99),
        "gc_events": gc_count,
        "total_samples": len(data),
    }


def normalize_timestamps(data: list[dict]) -> list[float]:
    if not data:
        return []
    start_time = min(d["timestamp"] for d in data)
    return [d["timestamp"] - start_time for d in data]


def create_memory_plot(
    with_gc: list[dict], without_gc: list[dict], output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    if with_gc:
        times_wg = normalize_timestamps(with_gc)
        mem_wg = [d["mem"] for d in with_gc]
        ax.plot(
            times_wg,
            mem_wg,
            label="With NeuroGC",
            color="#a6e3a1",
            linewidth=1.5,
        )

    if without_gc:
        times_wog = normalize_timestamps(without_gc)
        mem_wog = [d["mem"] for d in without_gc]
        ax.plot(
            times_wog,
            mem_wog,
            label="Without NeuroGC",
            color="#f38ba8",
            linewidth=1.5,
        )

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Memory Usage (%)", fontsize=12)
    ax.set_title("Memory Usage Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#1e1e2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#1e1e2e")
    plt.close()


def create_latency_plot(
    with_gc: list[dict], without_gc: list[dict], output_path: Path
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # P95 Latency
    if with_gc:
        times_wg = normalize_timestamps(with_gc)
        p95_wg = [d["p95"] for d in with_gc]
        ax1.plot(
            times_wg,
            p95_wg,
            label="With NeuroGC",
            color="#a6e3a1",
            linewidth=1.5,
        )

    if without_gc:
        times_wog = normalize_timestamps(without_gc)
        p95_wog = [d["p95"] for d in without_gc]
        ax1.plot(
            times_wog,
            p95_wog,
            label="Without NeuroGC",
            color="#f38ba8",
            linewidth=1.5,
        )

    ax1.set_xlabel("Time (seconds)", fontsize=12)
    ax1.set_ylabel("P95 Latency (ms)", fontsize=12)
    ax1.set_title("P95 Latency Comparison", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # P99 Latency
    if with_gc:
        p99_wg = [d["p99"] for d in with_gc]
        ax2.plot(
            times_wg,
            p99_wg,
            label="With NeuroGC",
            color="#a6e3a1",
            linewidth=1.5,
        )

    if without_gc:
        p99_wog = [d["p99"] for d in without_gc]
        ax2.plot(
            times_wog,
            p99_wog,
            label="Without NeuroGC",
            color="#f38ba8",
            linewidth=1.5,
        )

    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("P99 Latency (ms)", fontsize=12)
    ax2.set_title("P99 Latency Comparison", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Style both axes
    for ax in [ax1, ax2]:
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="white")

    fig.patch.set_facecolor("#1e1e2e")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#1e1e2e")
    plt.close()


def create_gc_events_plot(
    with_gc: list[dict], without_gc: list[dict], output_path: Path
) -> None:
    """Create GC events timeline plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if with_gc:
        times_wg = normalize_timestamps(with_gc)
        gc_times_wg = [
            t for t, d in zip(times_wg, with_gc) if d["gc_triggered"]
        ]
        for t in gc_times_wg:
            ax.axvline(x=t, color="#a6e3a1", alpha=0.7, linewidth=1)
        # Add dummy line for legend
        ax.axvline(
            x=-100,
            color="#a6e3a1",
            alpha=0.7,
            linewidth=2,
            label="With NeuroGC",
        )

    if without_gc:
        times_wog = normalize_timestamps(without_gc)
        gc_times_wog = [
            t for t, d in zip(times_wog, without_gc) if d["gc_triggered"]
        ]
        for t in gc_times_wog:
            ax.axvline(
                x=t, color="#f38ba8", alpha=0.7, linewidth=1, linestyle="--"
            )
        ax.axvline(
            x=-100,
            color="#f38ba8",
            alpha=0.7,
            linewidth=2,
            linestyle="--",
            label="Without NeuroGC",
        )

    # Set x-axis limits
    all_times = []
    if with_gc:
        all_times.extend(normalize_timestamps(with_gc))
    if without_gc:
        all_times.extend(normalize_timestamps(without_gc))
    if all_times:
        ax.set_xlim(0, max(all_times))

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("GC Events", fontsize=12)
    ax.set_title("GC Event Timeline", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_yticks([])

    # Style
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#1e1e2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#1e1e2e")
    plt.close()


def create_rps_plot(
    with_gc: list[dict], without_gc: list[dict], output_path: Path
) -> None:
    """Create RPS (requests per second) timeline plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if with_gc:
        times_wg = normalize_timestamps(with_gc)
        rps_wg = [d["rps"] for d in with_gc]
        ax.plot(
            times_wg,
            rps_wg,
            label="With NeuroGC",
            color="#a6e3a1",
            linewidth=1.5,
        )

    if without_gc:
        times_wog = normalize_timestamps(without_gc)
        rps_wog = [d["rps"] for d in without_gc]
        ax.plot(
            times_wog,
            rps_wog,
            label="Without NeuroGC",
            color="#f38ba8",
            linewidth=1.5,
        )

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Requests per Second", fontsize=12)
    ax.set_title("RPS Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Style
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#1e1e2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#1e1e2e")
    plt.close()


def calculate_improvement(
    with_val: float, without_val: float, lower_is_better: bool = True
) -> str:
    if without_val == 0:
        return "N/A"

    if lower_is_better:
        diff = without_val - with_val
        pct = (diff / without_val) * 100
    else:
        diff = with_val - without_val
        pct = (diff / without_val) * 100

    if abs(pct) < 0.05:
        return "0.0%"

    if pct > 0:
        return f"🟢 +{pct:.1f}%"
    else:
        return f"🔴 {pct:.1f}%"


def generate_readme(
    output_dir: Path,
    stats_with_gc: dict,
    stats_without_gc: dict,
    model_metadata: dict,
    system_info: dict,
    benchmark_date: datetime,
    csv_filename: str,
) -> None:
    """Generate README.md for the benchmark."""
    readme_content = f"""# Benchmark Results

**Date:** {benchmark_date.strftime("%B %d, %Y at %H:%M")}

**Raw Data:** [{csv_filename}](./{csv_filename})

## Performance Summary

| Metric | Without NeuroGC | With NeuroGC | Improvement |
| ------ | --------------- | ------------ | ----------- |
| Avg CPU (%) | {stats_without_gc.get("avg_cpu", 0):.1f} | {stats_with_gc.get("avg_cpu", 0):.1f} | {calculate_improvement(stats_with_gc.get("avg_cpu", 0), stats_without_gc.get("avg_cpu", 0), True)} |
| Avg Memory (%) | {stats_without_gc.get("avg_mem", 0):.1f} | {stats_with_gc.get("avg_mem", 0):.1f} | {calculate_improvement(stats_with_gc.get("avg_mem", 0), stats_without_gc.get("avg_mem", 0), True)} |
| Avg Disk Read | {stats_without_gc.get("avg_disk_read", 0):.2f} | {stats_with_gc.get("avg_disk_read", 0):.2f} | {calculate_improvement(stats_with_gc.get("avg_disk_read", 0), stats_without_gc.get("avg_disk_read", 0), True)} |
| Avg Disk Write | {stats_without_gc.get("avg_disk_write", 0):.2f} | {stats_with_gc.get("avg_disk_write", 0):.2f} | {calculate_improvement(stats_with_gc.get("avg_disk_write", 0), stats_without_gc.get("avg_disk_write", 0), True)} |
| Avg Net Sent | {stats_without_gc.get("avg_net_sent", 0):.2f} | {stats_with_gc.get("avg_net_sent", 0):.2f} | {calculate_improvement(stats_with_gc.get("avg_net_sent", 0), stats_without_gc.get("avg_net_sent", 0), True)} |
| Avg Net Recv | {stats_without_gc.get("avg_net_recv", 0):.2f} | {stats_with_gc.get("avg_net_recv", 0):.2f} | {calculate_improvement(stats_with_gc.get("avg_net_recv", 0), stats_without_gc.get("avg_net_recv", 0), True)} |
| P95 Latency (ms) | {stats_without_gc.get("avg_p95", 0):.1f} | {stats_with_gc.get("avg_p95", 0):.1f} | {calculate_improvement(stats_with_gc.get("avg_p95", 0), stats_without_gc.get("avg_p95", 0), True)} |
| P99 Latency (ms) | {stats_without_gc.get("avg_p99", 0):.1f} | {stats_with_gc.get("avg_p99", 0):.1f} | {calculate_improvement(stats_with_gc.get("avg_p99", 0), stats_without_gc.get("avg_p99", 0), True)} |
| Avg RPS | {stats_without_gc.get("avg_rps", 0):.1f} | {stats_with_gc.get("avg_rps", 0):.1f} | {calculate_improvement(stats_with_gc.get("avg_rps", 0), stats_without_gc.get("avg_rps", 0), False)} |
| GC Events | {stats_without_gc.get("gc_events", 0)} | {stats_with_gc.get("gc_events", 0)} | {calculate_improvement(stats_with_gc.get("gc_events", 0), stats_without_gc.get("gc_events", 0), False)} |


## Visualizations

### Memory Usage Comparison

![Memory Usage](./memory_usage.png)

### Latency Comparison

![Latency Comparison](./latency_comparison.png)

### GC Event Timeline

![GC Events](./gc_events.png)

### RPS Over Time

![RPS Timeline](./rps_timeline.png)

## ML Model Metadata

| Property | Value |
| -------- | ----- |
| Model Name | {model_metadata.get("name", "N/A")} |
| Version | {model_metadata.get("version", "N/A")} |
| Description | {model_metadata.get("description", "N/A")} |
| Input Features | {", ".join(model_metadata.get("input_features", []))} |
| Sequence Length | {model_metadata.get("sequence_length", "N/A")} |

## System Information

| Property | Value |
| -------- | ----- |
| Operating System | {system_info.get("os", "N/A")} |
| Architecture | {system_info.get("architecture", "N/A")} |
| CPU | {system_info.get("cpu_model", "N/A")} |
| CPU Cores | {system_info.get("cpu_cores", "N/A")} (logical: {system_info.get("cpu_cores_logical", "N/A")}) |
| Memory | {system_info.get("memory_gb", "N/A")} GB |
| Disk | {system_info.get("disk_gb", "N/A")} GB |
| Python Version | {system_info.get("python_version", "N/A")} |

## Benchmark Details

| Property | Value |
| -------- | ----- |
| Total Samples (with GC) | {stats_with_gc.get("total_samples", 0)} |
| Total Samples (without GC) | {stats_without_gc.get("total_samples", 0)} |
| Duration | ~{max(stats_with_gc.get("total_samples", 0), stats_without_gc.get("total_samples", 0))} seconds |
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"Generated README: {readme_path}")


def analyze_benchmark(
    csv_path: str,
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Analyze a benchmark CSV file and generate visualizations and README.

    Args:
        csv_path: Path to the benchmark.csv file
        model_name: Name of the ML model used (auto-detected if not provided)
        output_dir: Output directory (auto-generated if not provided)

    Returns:
        Path to the output directory
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {csv_path}")

    # Load config for defaults
    config = load_config()

    # Determine model name
    if model_name is None:
        # Try auto-detection from checkpoint
        model_name = detect_model_from_checkpoint()
        if model_name is None:
            # Fall back to config
            model_name = config.get("default_model", "unknown")
        print(f"Auto-detected model: {model_name}")

    # Determine output directory
    benchmark_date = datetime.now()
    if output_dir is None:
        timestamp = benchmark_date.strftime("%d-%m-%Y-%H-%M")
        output_dir = Path("benchmarks") / model_name / timestamp
    else:
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Copy CSV to output directory
    csv_dest = output_dir / "benchmark.csv"
    if csv_path.resolve() != csv_dest.resolve():
        shutil.copy(csv_path, csv_dest)
        print(f"Copied CSV to: {csv_dest}")

    # Load and parse data
    print("Loading benchmark data...")
    data = load_benchmark_csv(str(csv_path))
    with_gc, without_gc = split_by_server(data)

    print(f"  With GC samples: {len(with_gc)}")
    print(f"  Without GC samples: {len(without_gc)}")

    # Calculate statistics
    stats_with_gc = calculate_stats(with_gc)
    stats_without_gc = calculate_stats(without_gc)

    # Get model metadata and system info
    model_metadata = get_model_metadata(model_name)
    system_info = get_system_info()

    # Generate visualizations
    print("Generating visualizations...")
    create_memory_plot(with_gc, without_gc, output_dir / "memory_usage.png")
    print("  - memory_usage.png")

    create_latency_plot(
        with_gc, without_gc, output_dir / "latency_comparison.png"
    )
    print("  - latency_comparison.png")

    create_gc_events_plot(with_gc, without_gc, output_dir / "gc_events.png")
    print("  - gc_events.png")

    create_rps_plot(with_gc, without_gc, output_dir / "rps_timeline.png")
    print("  - rps_timeline.png")

    # Generate README
    print("Generating README...")
    generate_readme(
        output_dir,
        stats_with_gc,
        stats_without_gc,
        model_metadata,
        system_info,
        benchmark_date,
        "benchmark.csv",
    )

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NeuroGC benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_benchmark.py benchmark.csv
  python analyze_benchmark.py benchmark.csv --model lstm
  python analyze_benchmark.py benchmark.csv --model transformer --output benchmarks/custom
        """,
    )
    parser.add_argument(
        "csv_path",
        help="Path to the benchmark CSV file",
    )
    parser.add_argument(
        "--model",
        "-m",
        dest="model_name",
        help="ML model name (auto-detected from gc_model.pth or config.json if not specified)",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_dir",
        help="Output directory (auto-generated as benchmarks/{model}/{date-time} if not specified)",
    )

    args = parser.parse_args()

    try:
        analyze_benchmark(
            csv_path=args.csv_path,
            model_name=args.model_name,
            output_dir=args.output_dir,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing benchmark: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
