import pandas as pd
import time
import joblib
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Define paths
base_dir = r"./"
train_data_path = os.path.join(base_dir, "train_data.csv")
test_data_path = os.path.join(base_dir, "test_data.csv")
models_dir = os.path.join(base_dir, "models")
results_path = os.path.join(base_dir, "evaluation_results.json")

# Create models directory if not exists
os.makedirs(models_dir, exist_ok=True)

def load_data():
    print("Loading datasets...")
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    
    # Separate features and labels
    # Drop 'url' as it's not a numeric feature for the model
    # Drop 'label' for X
    X_train = train_df.drop(columns=['url', 'label'])
    y_train = train_df['label']
    
    X_test = test_df.drop(columns=['url', 'label'])
    y_test = test_df['label']
    
    print(f"Features loaded: {list(X_train.columns)}")
    return X_train, y_train, X_test, y_test

def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test):
    print(f"\nTraining {model_name}...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1_Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A",
        "Training_Time_Sec": training_time
    }
    
    print(f"Done. Training time: {training_time:.4f}s")
    for k, v in metrics.items():
        if k != "Model":
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            
    # Save model
    model_path = os.path.join(models_dir, f"{model_name.replace(' ', '_')}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return metrics

def main():
    X_train, y_train, X_test, y_test = load_data()
    
    results = []
    
    # 1. Baseline: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_metrics = train_and_evaluate(lr_model, "Logistic Regression", X_train, y_train, X_test, y_test)
    results.append(lr_metrics)
    
    # 2. Main Model: XGBoost
    # XGBoost is naturally "lightweight" and fast, but we use default mostly.
    # We can tune n_estimators or max_depth if needed, but defaults usually beat LR easily.
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_metrics = train_and_evaluate(xgb_model, "XGBoost", X_train, y_train, X_test, y_test)
    results.append(xgb_metrics)
    
    # Save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nAll results saved to {results_path}")

if __name__ == "__main__":
    main()
