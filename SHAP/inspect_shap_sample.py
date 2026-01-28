import pandas as pd
import joblib
import os
import shap

# Define paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data")
test_data_path = os.path.join(data_dir, "test_data.csv")
model_path = os.path.join(base_dir, "models", "XGBoost.joblib")

def inspect_sample():
    print("Loading test data...")
    test_df = pd.read_csv(test_data_path)
    # Replicate the sampling
    subset = test_df.sample(n=1000, random_state=42)
    
    # Find the first Malicious sample (Label 0)
    malicious_subset = subset[subset['label'] == 0]
    if malicious_subset.empty:
        print("No malicious samples found in subset.")
        return

    sample_row = malicious_subset.iloc[0]
    
    print("\n--- Sample Details for Waterfall Plot ---")
    print(f"URL: {sample_row['url']}")
    print(f"True Label: {sample_row['label']} (0=Malicious, 1=Benign)")
    
    # Load model to get prediction
    model = joblib.load(model_path)
    X_sample = pd.DataFrame([sample_row.drop(['url', 'label'])])
    
    pred = model.predict(X_sample)[0]
    print(f"Predicted Label: {pred}")
    
    # Explain this prediction
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    print("\nTop Contributing Features for this sample:")
    # shap_values.values is (1, n_features)
    values = shap_values.values[0]
    feature_names = X_sample.columns
    
    # Pair and sort
    contributions = list(zip(feature_names, values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for name, val in contributions[:5]:
        feature_val = sample_row[name]
        print(f"Feature: {name}, Value: {feature_val}, SHAP: {val:.4f}")

if __name__ == "__main__":
    inspect_sample()
