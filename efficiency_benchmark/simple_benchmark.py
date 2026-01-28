import time
import joblib
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import textwrap
import numpy as np
import seaborn as sns

# Define paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data")
models_dir = os.path.join(base_dir, "models")
test_data_path = os.path.join(data_dir, "test_data.csv")
figures_dir = os.path.join(base_dir, "figures")
benchmark_results_path = os.path.join(base_dir, "benchmark_results.json")

# Cloud Server Specs (Requested by User)
SERVER_SPECS = {
    "System": "Ubuntu 24.04",
    "CPU": "2 vCPU",
    "Memory": "2 GiB",
    "Disk": "40 GiB",
    "Bandwidth": "200 Mbps (Peak)"
}

def benchmark_model(model_name, model_file, X_test, warmup_rounds=50, repeats=500, trim_count=10):
    print(f"\nBenchmarking {model_name}...")
    model_path = os.path.join(models_dir, model_file)
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return None

    # Load model
    model = joblib.load(model_path)
    
    samples = X_test.sample(n=1000, random_state=42)
    n_samples = len(samples)

    for _ in range(int(warmup_rounds)):
        _ = model.predict(samples)

    times = []
    for _ in range(int(repeats)):
        start_time = time.perf_counter()
        _ = model.predict(samples)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    times_sorted = sorted(times)
    trim = int(trim_count)
    if trim * 2 >= len(times_sorted):
        trim = max(0, (len(times_sorted) - 1) // 2)

    trimmed = times_sorted[trim:len(times_sorted) - trim] if trim > 0 else times_sorted

    times_np = np.asarray(times, dtype=float)
    trimmed_np = np.asarray(trimmed, dtype=float)

    mean_total = float(np.mean(times_np))
    std_total = float(np.std(times_np, ddof=1)) if len(times_np) > 1 else 0.0
    min_total = float(np.min(times_np))
    max_total = float(np.max(times_np))

    trimmed_mean_total = float(np.mean(trimmed_np))
    trimmed_std_total = float(np.std(trimmed_np, ddof=1)) if len(trimmed_np) > 1 else 0.0

    tps_runs = n_samples / times_np
    lat_ms_runs = (times_np * 1000.0) / n_samples
    tps_trimmed_runs = n_samples / trimmed_np
    lat_ms_trimmed_runs = (trimmed_np * 1000.0) / n_samples

    tps_mean = float(np.mean(tps_runs))
    tps_std = float(np.std(tps_runs, ddof=1)) if len(tps_runs) > 1 else 0.0
    lat_ms_mean = float(np.mean(lat_ms_runs))
    lat_ms_std = float(np.std(lat_ms_runs, ddof=1)) if len(lat_ms_runs) > 1 else 0.0

    tps_trimmed_mean = float(np.mean(tps_trimmed_runs))
    tps_trimmed_std = float(np.std(tps_trimmed_runs, ddof=1)) if len(tps_trimmed_runs) > 1 else 0.0
    lat_ms_trimmed_mean = float(np.mean(lat_ms_trimmed_runs))
    lat_ms_trimmed_std = float(np.std(lat_ms_trimmed_runs, ddof=1)) if len(lat_ms_trimmed_runs) > 1 else 0.0

    def _cv(mean_v, std_v):
        return float(std_v / mean_v) if mean_v not in (0.0, -0.0) else 0.0

    def _ci95(mean_v, std_v, n):
        if n <= 1:
            return 0.0
        return float(1.96 * (std_v / (n ** 0.5)))

    print(f"Warm-up rounds: {warmup_rounds}")
    print(f"Repetitions: {repeats}")
    print(f"Trimmed mean: drop {trim} fastest and {trim} slowest")
    print("\n[Total Time for 1000 samples]")
    print(f"Mean:   {mean_total:.6f}s  ±Std: {std_total:.6f}s  (CV={_cv(mean_total, std_total):.4%}, 95%CI=±{_ci95(mean_total, std_total, len(times_np)):.6f}s)")
    print(f"Min:    {min_total:.6f}s")
    print(f"Max:    {max_total:.6f}s")
    print(f"Trimmed Mean: {trimmed_mean_total:.6f}s  ±Std: {trimmed_std_total:.6f}s  (CV={_cv(trimmed_mean_total, trimmed_std_total):.4%}, 95%CI=±{_ci95(trimmed_mean_total, trimmed_std_total, len(trimmed_np)):.6f}s)")

    print("\n[TPS]")
    print(f"Mean:   {tps_mean:.2f}  ±Std: {tps_std:.2f}  (CV={_cv(tps_mean, tps_std):.4%}, 95%CI=±{_ci95(tps_mean, tps_std, len(tps_runs)):.2f})")
    print(f"Trimmed Mean: {tps_trimmed_mean:.2f}  ±Std: {tps_trimmed_std:.2f}  (CV={_cv(tps_trimmed_mean, tps_trimmed_std):.4%}, 95%CI=±{_ci95(tps_trimmed_mean, tps_trimmed_std, len(tps_trimmed_runs)):.2f})")

    print("\n[Avg Latency (ms)]")
    print(f"Mean:   {lat_ms_mean:.6f} ms  ±Std: {lat_ms_std:.6f} ms  (CV={_cv(lat_ms_mean, lat_ms_std):.4%}, 95%CI=±{_ci95(lat_ms_mean, lat_ms_std, len(lat_ms_runs)):.6f} ms)")
    print(f"Trimmed Mean: {lat_ms_trimmed_mean:.6f} ms  ±Std: {lat_ms_trimmed_std:.6f} ms  (CV={_cv(lat_ms_trimmed_mean, lat_ms_trimmed_std):.4%}, 95%CI=±{_ci95(lat_ms_trimmed_mean, lat_ms_trimmed_std, len(lat_ms_trimmed_runs)):.6f} ms)")

    return {
        "Model": model_name,
        "Warmup_Rounds": int(warmup_rounds),
        "Repeats": int(repeats),
        "Trim_Count": int(trim),
        "Total_Time_1000_Samples_Mean": mean_total,
        "Total_Time_1000_Samples_Std": std_total,
        "Total_Time_1000_Samples_Min": min_total,
        "Total_Time_1000_Samples_Max": max_total,
        "Total_Time_1000_Samples_TrimmedMean": trimmed_mean_total,
        "Total_Time_1000_Samples_TrimmedStd": trimmed_std_total,
        "TPS_Mean": tps_mean,
        "TPS_Std": tps_std,
        "TPS_TrimmedMean": tps_trimmed_mean,
        "TPS_TrimmedStd": tps_trimmed_std,
        "Avg_Latency_ms_Mean": lat_ms_mean,
        "Avg_Latency_ms_Std": lat_ms_std,
        "Avg_Latency_ms_TrimmedMean": lat_ms_trimmed_mean,
        "Avg_Latency_ms_TrimmedStd": lat_ms_trimmed_std,
        "Runs_Total_Time_Sec": times,
        "Total_Time_1000_Samples": trimmed_mean_total,
        "TPS": tps_trimmed_mean,
        "Avg_Latency_ms": lat_ms_trimmed_mean,
    }

def generate_performance_chart(results):
    times_sec = np.asarray(results.get("Runs_Total_Time_Sec", []), dtype=float)
    if times_sec.size == 0:
        print("No run timing data found; skipping chart generation.")
        return

    batch_ms = times_sec * 1000.0
    raw_mean_ms = float(results.get("Total_Time_1000_Samples_Mean", float(np.mean(times_sec))) * 1000.0)
    trimmed_mean_ms = float(results.get("Total_Time_1000_Samples_TrimmedMean", float(np.mean(times_sec))) * 1000.0)

    tps_mean = float(results.get("TPS_Mean", 0.0))
    tps_std = float(results.get("TPS_Std", 0.0))
    tps_trimmed_mean = float(results.get("TPS_TrimmedMean", results.get("TPS", 0.0)))
    tps_trimmed_std = float(results.get("TPS_TrimmedStd", 0.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    plt.suptitle("Model Performance Benchmark on Cloud Environment", fontsize=18, fontweight='bold', y=0.98)

    x_left, x_right = 1.8, 3.5
    dense_batch_ms = batch_ms[(batch_ms >= x_left) & (batch_ms <= x_right)]
    sns.histplot(
        dense_batch_ms,
        bins=45,
        kde=True,
        color="skyblue",
        edgecolor="white",
        alpha=0.85,
        ax=ax1,
    )
    ax1.set_xlim(x_left, x_right)
    ax1.set_title("A. Latency Distribution (Reveal Noise)", fontsize=15, fontweight='bold')
    ax1.set_xlabel("Processing Time per Batch (ms)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=13, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=11)

    ax1.axvline(trimmed_mean_ms, color="#006400", linestyle="-", linewidth=2.2, label=f"{trimmed_mean_ms:.2f} ms (Robust)")
    ax1.axvline(raw_mean_ms, color="#d62728", linestyle="--", linewidth=2.2, label=f"{raw_mean_ms:.2f} ms (Skewed)")

    max_ms = float(np.max(batch_ms))
    y_top = ax1.get_ylim()[1]
    ax1.annotate(
        f"Extreme Outlier: {max_ms:.1f} ms",
        xy=(x_right - 0.02, y_top * 0.78),
        xytext=(x_right - 0.55, y_top * 0.88),
        fontsize=11,
        fontweight='bold',
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        ha="left",
        va="center",
    )

    leg1 = ax1.legend(fontsize=11, loc="upper left")
    for t in leg1.get_texts():
        t.set_fontweight('bold')
    for t in ax1.get_xticklabels() + ax1.get_yticklabels():
        t.set_fontweight('bold')

    labels = ["Raw Mean", "Trimmed Mean"]
    values = [tps_mean, tps_trimmed_mean]
    errors = [tps_std, tps_trimmed_std]
    colors = ["#9e9e9e", "#1f77b4"]

    bars = ax2.bar(labels, values, yerr=errors, capsize=8, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_title("B. Throughput Stability (TPS)", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Metric Method", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Transactions Per Second (TPS)", fontsize=13, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=11)
    ax2.set_ylim(300000, max(values[0] + errors[0], values[1] + errors[1]) * 1.08)

    for rect, v in zip(bars, values):
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + ax2.get_ylim()[1] * 0.01,
            f"{v/1000.0:.0f}k",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight='bold',
        )
    for t in ax2.get_xticklabels() + ax2.get_yticklabels():
        t.set_fontweight('bold')

    # Add Server Specs as a text box
    specs_text = "Server Configuration:\n" + "\n".join([f"{k}: {v}" for k, v in SERVER_SPECS.items()])
    props = dict(boxstyle='round,pad=0.15', facecolor='wheat', alpha=0.5)
    
    # Place text box in the figure (outside subplots)
    fig.text(
        0.015,
        0.935,
        specs_text,
        fontsize=9,
        fontweight='bold',
        verticalalignment='top',
        horizontalalignment='left',
        bbox=props,
        family='monospace',
    )
    
    # Adjust layout to make room for text
    plt.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.12, wspace=0.32)
    
    # Save
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        
    output_path = os.path.join(figures_dir, "benchmark_performance_specs.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance chart saved to {output_path}")

def main():
    print("Loading test data for benchmarking...")
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(columns=['url', 'label'])
    
    # Benchmark XGBoost
    xgb_results = benchmark_model("XGBoost", "XGBoost.joblib", X_test, warmup_rounds=50, repeats=500, trim_count=10)
    
    if xgb_results:
        # Generate Chart with Specs
        generate_performance_chart(xgb_results)
        
        # Save results JSON
        with open(benchmark_results_path, 'w') as f:
            json.dump([xgb_results], f, indent=4)
        print(f"Benchmark results saved to {benchmark_results_path}")

if __name__ == "__main__":
    main()
