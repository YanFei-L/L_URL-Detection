import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import xgboost
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Define paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data")
test_data_path = os.path.join(data_dir, "test_data.csv")
evaluation_results_path = os.path.join(base_dir, "evaluation_results.json")
benchmark_results_path = os.path.join(base_dir, "benchmark_results.json")
models_dir = os.path.join(base_dir, "models")
figures_dir = os.path.join(base_dir, "figures")
model_path = os.path.join(models_dir, "XGBoost.joblib")

# Cloud Server Specs
SERVER_SPECS = {
    "System": "Ubuntu 24.04",
    "CPU": "2 vCPU",
    "Memory": "2 GiB",
    "Disk": "40 GiB",
    "Bandwidth": "200 Mbps (Peak)",
}

# Create figures directory
os.makedirs(figures_dir, exist_ok=True)

def load_data_and_model():
    print("Loading test data...")
    test_df = pd.read_csv(test_data_path)
    per_class = 500
    if 'label' not in test_df.columns or 'url' not in test_df.columns:
        raise KeyError("test_data.csv must contain 'url' and 'label' columns")

    df_mal = test_df[test_df['label'] == 0]
    df_ben = test_df[test_df['label'] == 1]

    sample_mal = df_mal.sample(n=min(per_class, len(df_mal)), random_state=42)
    sample_ben = df_ben.sample(n=min(per_class, len(df_ben)), random_state=42)
    sample_df = pd.concat([sample_mal, sample_ben], axis=0)

    if len(sample_df) < 1000:
        remaining = test_df.drop(index=sample_df.index, errors='ignore')
        fill_n = 1000 - len(sample_df)
        sample_df = pd.concat([sample_df, remaining.sample(n=fill_n, random_state=42)], axis=0)
    elif len(sample_df) > 1000:
        sample_df = sample_df.sample(n=1000, random_state=42)

    sample_df = sample_df.sample(frac=1.0, random_state=42)
    X_test = sample_df.drop(columns=['url', 'label'])
    y_test = sample_df['label']
    urls = sample_df['url']
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    return test_df, X_test, y_test, urls, model

def _load_reported_xgboost_metrics():
    if not os.path.exists(evaluation_results_path):
        print(f"[VERIFY] evaluation_results.json not found at: {evaluation_results_path}")
        return None

    try:
        with open(evaluation_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"[VERIFY] Failed to read evaluation_results.json: {e}")
        return None

    if not isinstance(results, list):
        print("[VERIFY] evaluation_results.json format is not a list. Skipping verification.")
        return None

    xgb_row = next((r for r in results if r.get('Model') == 'XGBoost'), None)
    if xgb_row is None:
        print("[VERIFY] No 'XGBoost' entry found in evaluation_results.json. Skipping verification.")
        return None

    return xgb_row

def verify_evaluation_vs_current_confusion_matrix(model, test_df):
    X_full = test_df.drop(columns=['url', 'label'])
    y_true = test_df['label'].astype(int).to_numpy()

    if not hasattr(model, 'predict_proba'):
        print("[VERIFY] Model has no predict_proba(); cannot compute AUC. Skipping verification.")
        return

    y_prob_pos1 = model.predict_proba(X_full)[:, 1]
    y_pred = (y_prob_pos1 >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    auc_val = roc_auc_score(y_true, y_prob_pos1)

    print("\n[VERIFY] Recomputed metrics on current test_data.csv + current XGBoost model")
    print("[VERIFY] Positive label for metrics/AUC: 1 (Benign)")
    print(f"[VERIFY] Test samples: {len(y_true)}")
    print(f"[VERIFY] Confusion matrix (rows=true [0,1], cols=pred [0,1]):\n{cm}")
    print(f"[VERIFY] Accuracy:  {acc:.10f}")
    print(f"[VERIFY] Precision: {prec:.10f}")
    print(f"[VERIFY] Recall:    {rec:.10f}")
    print(f"[VERIFY] F1_Score:  {f1:.10f}")
    print(f"[VERIFY] AUC:       {auc_val:.10f}\n")

    reported = _load_reported_xgboost_metrics()
    if reported is None:
        return

    def _compare(key, computed, atol=1e-6, rtol=1e-4):
        if key not in reported:
            print(f"[VERIFY] Missing key in evaluation_results.json: {key}")
            return
        try:
            reported_val = float(reported[key])
        except Exception:
            print(f"[VERIFY] Non-numeric value for key '{key}' in evaluation_results.json: {reported[key]}")
            return

        diff = abs(reported_val - computed)
        tol = max(atol, rtol * max(1.0, abs(reported_val)))
        status = "OK" if diff <= tol else "MISMATCH"
        print(f"[VERIFY] {status:8s} {key}: reported={reported_val:.10f}, computed={computed:.10f}, diff={diff:.10f}")

    print("[VERIFY] Comparing evaluation_results.json (XGBoost) vs recomputed metrics")
    _compare('Accuracy', acc)
    _compare('Precision', prec)
    _compare('Recall', rec)
    _compare('F1_Score', f1)
    _compare('AUC', auc_val)

def plot_shap_summary(shap_values, X_test):
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(11, 8.5))
    # Beeswarm plot
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Summary Plot (Global Feature Importance)", fontsize=16, fontweight='bold')
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=11)
    for t in ax.get_yticklabels() + ax.get_xticklabels():
        t.set_fontweight('bold')
    ax.set_xlabel(ax.get_xlabel(), fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "shap_summary_beeswarm.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

def merge_shap_summary_and_bar(
    beeswarm_filename="shap_summary_beeswarm.png",
    bar_filename="shap_importance_bar.png",
    output_filename="shap_beeswarm_importance_combined.png",
):
    beeswarm_path = os.path.join(figures_dir, beeswarm_filename)
    bar_path = os.path.join(figures_dir, bar_filename)
    if not os.path.exists(beeswarm_path) or not os.path.exists(bar_path):
        print("[MERGE] Missing input images; skipping merge.")
        print(f"[MERGE] beeswarm: {beeswarm_path}")
        print(f"[MERGE] bar:      {bar_path}")
        return

    beeswarm_img = plt.imread(beeswarm_path)
    bar_img = plt.imread(bar_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7.5))
    ax1.imshow(beeswarm_img)
    ax1.axis('off')
    ax1.set_title("SHAP Summary (Beeswarm)", fontsize=14, fontweight='bold')

    ax2.imshow(bar_img)
    ax2.axis('off')
    ax2.set_title("SHAP Feature Importance (Bar)", fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(figures_dir, output_filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[MERGE] Saved combined figure to {save_path}")

def crop_benchmark_left_panel(
    input_filename="benchmark_performance_specs.png",
    output_filename="benchmark_performance_specs_left.png",
):
    if os.path.exists(benchmark_results_path):
        try:
            with open(benchmark_results_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            results = payload[0] if isinstance(payload, list) and payload else None
        except Exception as e:
            results = None
            print(f"[BENCH-CROP] Failed to read benchmark_results.json: {e}")

        if isinstance(results, dict) and results.get('Runs_Total_Time_Sec'):
            times_sec = np.asarray(results.get('Runs_Total_Time_Sec', []), dtype=float)
            if times_sec.size > 0:
                batch_ms = times_sec * 1000.0
                raw_mean_ms = float(results.get('Total_Time_1000_Samples_Mean', float(np.mean(times_sec))) * 1000.0)
                trimmed_mean_ms = float(results.get('Total_Time_1000_Samples_TrimmedMean', float(np.mean(times_sec))) * 1000.0)

                def _kde_counts(x_grid: np.ndarray, data: np.ndarray, bin_width: float) -> np.ndarray:
                    if data.size == 0:
                        return np.zeros_like(x_grid)
                    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
                    bw = 1.06 * std * (data.size ** (-1.0 / 5.0)) if std > 0 else 1e-6
                    bw = max(bw, 1e-6)
                    z = (x_grid[:, None] - data[None, :]) / bw
                    density = np.mean(np.exp(-0.5 * z * z), axis=1) / (bw * np.sqrt(2.0 * np.pi))
                    return density * data.size * bin_width

                fig, ax = plt.subplots(figsize=(10.8, 6.0))
                fig.suptitle(
                    "Model Performance Benchmark on Cloud Environment",
                    fontsize=16,
                    fontweight='bold',
                    x=0.5,
                    y=0.97,
                )

                x_left, x_right = 1.8, 3.5
                dense_batch_ms = batch_ms[(batch_ms >= x_left) & (batch_ms <= x_right)]
                bins = 45
                _, bin_edges, _ = ax.hist(
                    dense_batch_ms,
                    bins=bins,
                    range=(x_left, x_right),
                    color='skyblue',
                    edgecolor='white',
                    alpha=0.85,
                )
                bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else (x_right - x_left) / bins
                x_grid = np.linspace(x_left, x_right, 400)
                kde_counts = _kde_counts(x_grid, dense_batch_ms, bin_width)
                ax.plot(x_grid, kde_counts, color='#1f77b4', linewidth=2.0)

                ax.set_xlim(x_left, x_right)
                ax.set_title("Latency Distribution (Reveal Noise)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Processing Time per Batch (ms)", fontsize=12, fontweight='bold')
                ax.set_ylabel("Count", fontsize=12, fontweight='bold')
                ax.tick_params(axis='both', labelsize=10)
                for t in ax.get_xticklabels() + ax.get_yticklabels():
                    t.set_fontweight('bold')

                ax.axvline(trimmed_mean_ms, color='#006400', linestyle='-', linewidth=2.2, label=f"{trimmed_mean_ms:.2f} ms (Robust)")
                ax.axvline(raw_mean_ms, color='#d62728', linestyle='--', linewidth=2.2, label=f"{raw_mean_ms:.2f} ms (Skewed)")

                max_ms = float(np.max(batch_ms))
                y_top = ax.get_ylim()[1]
                ax.annotate(
                    f"Extreme Outlier: {max_ms:.1f} ms",
                    xy=(x_right - 0.02, y_top * 0.75),
                    xytext=(x_right - 0.55, y_top * 0.88),
                    fontsize=10,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    ha='left',
                    va='center',
                )

                leg = ax.legend(fontsize=10, loc='upper left')
                for t in leg.get_texts():
                    t.set_fontweight('bold')

                specs_text = "Server Configuration:\n" + "\n".join([f"{k}: {v}" for k, v in SERVER_SPECS.items()])
                props = dict(boxstyle='round,pad=0.12', facecolor='wheat', alpha=0.5)
                fig.text(
                    0.015,
                    0.90,
                    specs_text,
                    fontsize=8,
                    fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=props,
                    family='monospace',
                )

                plt.subplots_adjust(left=0.22, right=0.98, top=0.82, bottom=0.14)
                output_path = os.path.join(figures_dir, output_filename)
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"[BENCH-CROP] Saved left-only benchmark figure to {output_path}")
                return

    input_path = os.path.join(figures_dir, input_filename)
    if not os.path.exists(input_path):
        print(f"[BENCH-CROP] Benchmark figure not found: {input_path}. Skipping.")
        return

    img = plt.imread(input_path)
    if img.ndim < 2:
        print(f"[BENCH-CROP] Unexpected image format: {input_path}. Skipping.")
        return

    h, w = img.shape[0], img.shape[1]
    rgb = img[:, :, :3] if img.ndim == 3 and img.shape[2] >= 3 else img
    gray = np.mean(rgb.astype(float), axis=2) if rgb.ndim == 3 else rgb.astype(float)
    ink = 1.0 - gray
    col_score = np.sum(ink, axis=0)

    start = int(w * 0.45)
    end = int(w * 0.65)
    if end <= start + 5:
        cut_col = w // 2
    else:
        cut_col = start + int(np.argmin(col_score[start:end]))

    cut_col = max(1, min(w, cut_col))
    cropped = img[:, :cut_col].copy()

    output_path = os.path.join(figures_dir, output_filename)
    plt.imsave(output_path, cropped)
    print(f"[BENCH-CROP] Saved left-only benchmark figure to {output_path}")

def plot_shap_bar(shap_values, X_test):
    print("Generating SHAP Bar Plot...")
    plt.figure(figsize=(11, 8.5))
    shap.plots.bar(shap_values, show=False)
    plt.title("Top Feature Importance (SHAP Bar Plot)", fontsize=16, fontweight='bold')
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=11)
    for t in ax.get_yticklabels() + ax.get_xticklabels():
        t.set_fontweight('bold')
    ax.set_xlabel(ax.get_xlabel(), fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "shap_importance_bar.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

def plot_shap_waterfall(shap_values, X_test, y_test, urls, explainer, model, target_label, output_filename):
    print(f"Generating SHAP Waterfall Plot (Local Interpretation) for label={target_label}...")
    
    # Find a sample that is predicted as Malicious (Class 0) or Benign (Class 1)
    # In this dataset: 0 is Malicious, 1 is Benign.
    # We want to show why a malicious URL was caught.
    # XGBoost prediction: higher value -> Class 1 (Benign). Lower value -> Class 0 (Malicious).
    # Wait, XGBoost predict_proba gives prob of class 1.
    # SHAP values correspond to the raw output (log odds) or probability depending on model_output.
    
    # Let's find a high confidence malicious example (low SHAP value sum)
    # But we need to be careful with indices. X_test is sampled.
    
    if not hasattr(model, 'predict_proba'):
        candidates = np.where(y_test.to_numpy() == target_label)[0]
        if len(candidates) == 0:
            print(f"[WATERFALL] No samples with label={target_label} found in SHAP subset; falling back to sample_idx=0")
            sample_idx = 0
        else:
            sample_idx = int(candidates[0])
        y_prob_pos1 = None
        y_pred = None
    else:
        y_prob_pos1 = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob_pos1 >= 0.5).astype(int)

        label_idx = np.where(y_test.to_numpy() == target_label)[0]
        if len(label_idx) == 0:
            print(f"[WATERFALL] No samples with label={target_label} found in SHAP subset; falling back to sample_idx=0")
            sample_idx = 0
        else:
            predicted_correct_idx = label_idx[y_pred[label_idx] == target_label]

            if len(predicted_correct_idx) > 0:
                if target_label == 0:
                    # Malicious case: prefer very low P(class=1)
                    sample_idx = int(predicted_correct_idx[np.argmin(y_prob_pos1[predicted_correct_idx])])
                else:
                    # Benign case: prefer very high P(class=1)
                    sample_idx = int(predicted_correct_idx[np.argmax(y_prob_pos1[predicted_correct_idx])])
            else:
                # Fallback: pick the most confident example among the true-label set
                if target_label == 0:
                    sample_idx = int(label_idx[np.argmin(y_prob_pos1[label_idx])])
                else:
                    sample_idx = int(label_idx[np.argmax(y_prob_pos1[label_idx])])

    try:
        selected_url = str(urls.iloc[sample_idx])
    except Exception:
        selected_url = "<unavailable>"

    if y_prob_pos1 is not None and y_pred is not None:
        print(
            "[WATERFALL] Selected sample: "
            f"idx={sample_idx}, url={selected_url}, true={int(y_test.iloc[sample_idx])}, "
            f"pred={int(y_pred[sample_idx])}, P(class=1|x)={y_prob_pos1[sample_idx]:.6f}"
        )
    else:
        print(f"[WATERFALL] Selected sample: idx={sample_idx}, url={selected_url}, true={int(y_test.iloc[sample_idx])}")
    
    plt.figure(figsize=(11, 6.8))
    # SHAP waterfall plot
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    if y_prob_pos1 is not None and y_pred is not None:
        plt.title(
            f"Local Explanation (Sample) idx={sample_idx}, true={int(y_test.iloc[sample_idx])}, pred={int(y_pred[sample_idx])}, P(class=1)={y_prob_pos1[sample_idx]:.4f}",
            fontsize=13,
            fontweight='bold',
        )
    else:
        plt.title(f"Local Explanation (Sample) idx={sample_idx}", fontsize=15, fontweight='bold')

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    save_path = os.path.join(figures_dir, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

def main():
    test_df, X_test, y_test, urls, model = load_data_and_model()

    verify_evaluation_vs_current_confusion_matrix(model, test_df)
    
    print("Initializing SHAP Explainer...")
    # XGBoost is a tree model, so we use TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    print("Computing SHAP values...")
    shap_values = explainer(X_test)
    
    # Print Top Features
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = X_test.columns
    feature_importance = pd.DataFrame(list(zip(feature_names, mean_shap)), columns=['feature', 'importance'])
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    print("\nTop 10 Important Features (by mean |SHAP|):")
    print(feature_importance.head(10))
    
    # Generate Plots
    plot_shap_summary(shap_values, X_test)
    plot_shap_bar(shap_values, X_test)
    merge_shap_summary_and_bar()
    plot_shap_waterfall(shap_values, X_test, y_test, urls, explainer, model, target_label=0, output_filename="shap_waterfall_sample.png")
    plot_shap_waterfall(shap_values, X_test, y_test, urls, explainer, model, target_label=1, output_filename="shap_waterfall_white_sample.png")
    crop_benchmark_left_panel()
    
    print("SHAP analysis complete!")

if __name__ == "__main__":
    main()
