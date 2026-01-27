import pandas as pd
import os
from sklearn.model_selection import train_test_split
from feature_extractor import FeatureExtractor
from tqdm import tqdm

# Define paths
base_dir = r"./"
white_list_path = os.path.join(base_dir, "white_list.csv")
block_list_path = os.path.join(base_dir, "block_list.csv")
train_data_path = os.path.join(base_dir, "train_data.csv")
test_data_path = os.path.join(base_dir, "test_data.csv")

def load_and_extract(file_path, label_value, extractor):
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)
    
    features_list = []
    
    # Using tqdm for progress bar if available
    for url in tqdm(df['url'], desc=f"Extracting features for label {label_value}"):
        features = extractor.extract_all_features(url)
        features['url'] = url  # Add URL to dataframe
        features['label'] = label_value
        features_list.append(features)
        
    return pd.DataFrame(features_list)

def main():
    extractor = FeatureExtractor()
    
    # Process White List (Label 1)
    df_white = load_and_extract(white_list_path, 1, extractor)
    
    # Process Block List (Label 0)
    df_block = load_and_extract(block_list_path, 0, extractor)
    
    # Combine datasets
    print("Combining datasets...")
    full_df = pd.concat([df_white, df_block], ignore_index=True)
    
    # Shuffle and Split
    print("Splitting into Train (70%) and Test (30%)...")
    # Using random_state=42 as per paper rules
    train_df, test_df = train_test_split(full_df, test_size=0.3, random_state=42, stratify=full_df['label'])
    
    # Save to CSV
    print(f"Saving training data to {train_data_path} ({len(train_df)} records)...")
    train_df.to_csv(train_data_path, index=False)
    
    print(f"Saving test data to {test_data_path} ({len(test_df)} records)...")
    test_df.to_csv(test_data_path, index=False)
    
    print("Feature extraction and data split complete!")

if __name__ == "__main__":
    main()
