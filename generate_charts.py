import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np

# Define paths
base_dir = r"./"
test_data_path = os.path.join(base_dir, "test_data.csv")
models_dir = os.path.join(base_dir, "models")
figures_dir = os.path.join(base_dir, "figures")

# Create figures directory
os.makedirs(figures_dir, exist_ok=True)

def load_data():
    print("Loading test data...")
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(columns=['url', 'label'])
    y_test = test_df['label']
    return X_test, y_test

def plot_roc_curve(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.tick_params(axis='both', labelsize=11)

    roc_data = {}
    style_map = {
        "Logistic Regression": {"color": "#1f77b4", "linestyle": "-", "lw": 2.0},
        "XGBoost": {"color": "#ff7f0e", "linestyle": "-", "lw": 2.2},
    }

    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_data[model_name] = (fpr, tpr, roc_auc)

            style = style_map.get(model_name, {})
            ax.plot(
                fpr,
                tpr,
                label=f"{model_name} (AUC = {roc_auc:.6f})",
                **style,
            )

    ax.plot([0, 1], [0, 1], color='navy', lw=1.8, linestyle='--', alpha=0.8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(True, alpha=0.3)
    legend = ax.legend(loc="lower right")
    for text in legend.get_texts():
        text.set_fontsize(11)
        text.set_fontweight('bold')

    axins = inset_axes(ax, width="45%", height="45%", loc='upper right', borderpad=1.2)
    for model_name, (fpr, tpr, roc_auc) in roc_data.items():
        style = style_map.get(model_name, {})
        axins.plot(fpr, tpr, **style)

    axins.set_xlim(0.0, 0.01)
    axins.set_ylim(0.98, 1.0)
    axins.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    axins.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axins.set_xticks([0.0, 0.005, 0.01])
    axins.tick_params(axis='x', labelsize=8)
    axins.tick_params(axis='y', labelsize=9)
    axins.grid(True, alpha=0.25)
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

    if "Logistic Regression" in roc_data and "XGBoost" in roc_data:
        fpr_lr, tpr_lr, _ = roc_data["Logistic Regression"]
        fpr_xgb, tpr_xgb, _ = roc_data["XGBoost"]
        grid = np.linspace(0.0, 0.01, 200)
        tpr_lr_i = np.interp(grid, fpr_lr, tpr_lr)
        tpr_xgb_i = np.interp(grid, fpr_xgb, tpr_xgb)
        delta = tpr_xgb_i - tpr_lr_i
        idx = int(np.argmax(np.abs(delta)))
        axins.text(
            0.00025,
            0.9805,
            f"max |ΔTPR|={abs(delta[idx]):.4f} @FPR={grid[idx]:.4f}",
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="0.8"),
        )

    fig.tight_layout()
    save_path = os.path.join(figures_dir, "roc_curve.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC Curve saved to {save_path}")
    plt.close(fig)

def plot_confusion_matrix(model, model_name, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Malicious (0)', 'Benign (1)'],
        yticklabels=['Malicious (0)', 'Benign (1)'],
        ax=ax,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=15, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight('bold')
    
    save_path = os.path.join(figures_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion Matrix for {model_name} saved to {save_path}")
    plt.close(fig)

def main():
    X_test, y_test = load_data()
    
    # Load models
    models = {}
    model_files = {
        "Logistic Regression": "Logistic_Regression.joblib",
        "XGBoost": "XGBoost.joblib"
    }
    
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            print(f"Loading {name} from {path}...")
            models[name] = joblib.load(path)
        else:
            print(f"Warning: Model file {path} not found.")
            
    if not models:
        print("No models loaded. Exiting.")
        return

    # Plot ROC Curve
    plot_roc_curve(models, X_test, y_test)
    
    # Plot Confusion Matrices
    for name, model in models.items():
        plot_confusion_matrix(model, name, X_test, y_test)

if __name__ == "__main__":
    main()
