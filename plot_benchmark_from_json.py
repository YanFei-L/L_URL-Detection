import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


base_dir = r"./"
figures_dir = os.path.join(base_dir, "figures")
benchmark_results_path = os.path.join(base_dir, "benchmark_results.json")
summary_txt_path = os.path.join(base_dir, "1.txt")

SERVER_SPECS = {
    "System": "Ubuntu 24.04",
    "CPU": "2 vCPU",
    "Memory": "2 GiB",
    "Disk": "40 GiB",
    "Bandwidth": "200 Mbps (Peak)",
}


def _kde_counts(x_grid: np.ndarray, data: np.ndarray, bin_width: float) -> np.ndarray:
    if data.size == 0:
        return np.zeros_like(x_grid)

    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    bw = 1.06 * std * (data.size ** (-1.0 / 5.0)) if std > 0 else 1e-6
    bw = max(bw, 1e-6)

    z = (x_grid[:, None] - data[None, :]) / bw
    density = np.mean(np.exp(-0.5 * z * z), axis=1) / (bw * np.sqrt(2.0 * np.pi))

    return density * data.size * bin_width


def _try_load_summary_numbers(txt_path: str) -> dict:
    if not os.path.exists(txt_path):
        return {}

    with open(txt_path, "r", encoding="utf-8") as f:
        txt = f.read()

    def _find(pattern: str):
        m = re.search(pattern, txt)
        return float(m.group(1)) if m else None

    return {
        "Total_Time_1000_Samples_Mean": _find(r"Mean:\s+(\d+\.\d+)s"),
        "Total_Time_1000_Samples_TrimmedMean": _find(r"Trimmed Mean:\s+(\d+\.\d+)s"),
        "TPS_Mean": _find(r"\[TPS\][\s\S]*?Mean:\s+(\d+\.\d+)"),
        "TPS_TrimmedMean": _find(r"\[TPS\][\s\S]*?Trimmed Mean:\s+(\d+\.\d+)"),
        "Avg_Latency_ms_Mean": _find(r"\[Avg Latency \(ms\)\][\s\S]*?Mean:\s+(\d+\.\d+)\s+ms"),
        "Avg_Latency_ms_TrimmedMean": _find(r"\[Avg Latency \(ms\)\][\s\S]*?Trimmed Mean:\s+(\d+\.\d+)\s+ms"),
    }


def _print_consistency_check(json_results: dict, summary_numbers: dict) -> None:
    if not summary_numbers:
        return

    keys = [
        "Total_Time_1000_Samples_Mean",
        "Total_Time_1000_Samples_TrimmedMean",
        "TPS_Mean",
        "TPS_TrimmedMean",
        "Avg_Latency_ms_Mean",
        "Avg_Latency_ms_TrimmedMean",
    ]

    for k in keys:
        a = json_results.get(k)
        b = summary_numbers.get(k)
        if a is None or b is None:
            continue

        tol = 1e-6 if "Time" in k or "Latency" in k else 1e-2
        if abs(float(a) - float(b)) > tol:
            print(f"[WARN] Summary mismatch for {k}: json={a} vs 1.txt={b}")


def plot_from_benchmark_json(
    json_path: str = benchmark_results_path,
    summary_path: str = summary_txt_path,
    output_path: str | None = None,
) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not payload:
        raise ValueError("benchmark_results.json is empty")

    results = payload[0]

    summary_numbers = _try_load_summary_numbers(summary_path)
    _print_consistency_check(results, summary_numbers)

    times_sec = np.asarray(results.get("Runs_Total_Time_Sec", []), dtype=float)
    if times_sec.size == 0:
        raise ValueError("Runs_Total_Time_Sec is missing or empty")

    batch_ms = times_sec * 1000.0

    raw_mean_ms = float(results.get("Total_Time_1000_Samples_Mean", float(np.mean(times_sec))) * 1000.0)
    trimmed_mean_ms = float(results.get("Total_Time_1000_Samples_TrimmedMean", float(np.mean(times_sec))) * 1000.0)

    tps_mean = float(results.get("TPS_Mean", 0.0))
    tps_std = float(results.get("TPS_Std", 0.0))
    tps_trimmed_mean = float(results.get("TPS_TrimmedMean", results.get("TPS", 0.0)))
    tps_trimmed_std = float(results.get("TPS_TrimmedStd", 0.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    plt.suptitle("Model Performance Benchmark on Cloud Environment", fontsize=18, fontweight="bold", y=0.98)

    x_left, x_right = 1.8, 3.5
    dense_batch_ms = batch_ms[(batch_ms >= x_left) & (batch_ms <= x_right)]
    bins = 45
    counts, bin_edges, _ = ax1.hist(
        dense_batch_ms,
        bins=bins,
        range=(x_left, x_right),
        color="skyblue",
        edgecolor="white",
        alpha=0.85,
    )

    bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else (x_right - x_left) / bins
    x_grid = np.linspace(x_left, x_right, 400)
    kde_counts = _kde_counts(x_grid, dense_batch_ms, bin_width)
    ax1.plot(x_grid, kde_counts, color="#1f77b4", linewidth=2.0, label="KDE")

    ax1.set_xlim(x_left, x_right)
    ax1.set_title("A. Latency Distribution (Reveal Noise)", fontsize=15, fontweight="bold")
    ax1.set_xlabel("Processing Time per Batch (ms)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=13, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=11)

    ax1.axvline(
        trimmed_mean_ms,
        color="#006400",
        linestyle="-",
        linewidth=2.2,
        label=f"{trimmed_mean_ms:.2f} ms (Robust)",
    )
    ax1.axvline(
        raw_mean_ms,
        color="#d62728",
        linestyle="--",
        linewidth=2.2,
        label=f"{raw_mean_ms:.2f} ms (Skewed)",
    )

    max_ms = float(np.max(batch_ms))
    y_top = ax1.get_ylim()[1]
    ax1.annotate(
        f"Extreme Outlier: {max_ms:.1f} ms",
        xy=(x_right - 0.02, y_top * 0.78),
        xytext=(x_right - 0.55, y_top * 0.88),
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        ha="left",
        va="center",
    )

    leg1 = ax1.legend(fontsize=11, loc="upper left")
    for t in leg1.get_texts():
        t.set_fontweight("bold")
    for t in ax1.get_xticklabels() + ax1.get_yticklabels():
        t.set_fontweight("bold")

    labels = ["Raw Mean", "Trimmed Mean"]
    values = [tps_mean, tps_trimmed_mean]
    errors = [tps_std, tps_trimmed_std]
    colors = ["#9e9e9e", "#1f77b4"]

    bars = ax2.bar(labels, values, yerr=errors, capsize=8, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_title("B. Throughput Stability (TPS)", fontsize=15, fontweight="bold")
    ax2.set_xlabel("Metric Method", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Transactions Per Second (TPS)", fontsize=13, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=11)

    y_upper = max(values[0] + errors[0], values[1] + errors[1]) * 1.08
    ax2.set_ylim(300000, y_upper)

    for rect, v in zip(bars, values):
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + y_upper * 0.01,
            f"{v/1000.0:.0f}k",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    for t in ax2.get_xticklabels() + ax2.get_yticklabels():
        t.set_fontweight("bold")

    specs_text = "Server Configuration:\n" + "\n".join([f"{k}: {v}" for k, v in SERVER_SPECS.items()])
    props = dict(boxstyle="round,pad=0.15", facecolor="wheat", alpha=0.5)
    fig.text(
        0.015,
        0.935,
        specs_text,
        fontsize=9,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=props,
        family="monospace",
    )

    plt.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.12, wspace=0.32)

    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    if output_path is None:
        output_path = os.path.join(figures_dir, "benchmark_performance_specs.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to {output_path}")
    return output_path


def main():
    plot_from_benchmark_json()


if __name__ == "__main__":
    main()
