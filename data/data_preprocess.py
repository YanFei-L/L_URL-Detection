import csv
import os

# Define paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data")
top1m_path = os.path.join(data_dir, "top-1m.csv")
verified_online_path = os.path.join(data_dir, "verified_online.csv")
white_list_path = os.path.join(data_dir, "white_list.csv")
block_list_path = os.path.join(data_dir, "block_list.csv")

TARGET_COUNT = 40000

def process_white_list():
    print(f"Processing white list from {top1m_path}...")
    data = []
    try:
        with open(top1m_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            count = 0
            for row in reader:
                if count >= TARGET_COUNT:
                    break
                if len(row) >= 2:
                    # top-1m.csv format: rank,domain
                    domain = row[1]
                    url = f"https://{domain}"
                    # Label 1 for White List
                    data.append([url, 1])
                    count += 1
        
        with open(white_list_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['url', 'label'])
            writer.writerows(data)
        print(f"Generated {white_list_path} with {len(data)} records.")
    except Exception as e:
        print(f"Error processing white list: {e}")

def process_block_list():
    print(f"Processing block list from {verified_online_path}...")
    data = []
    try:
        with open(verified_online_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                if count >= TARGET_COUNT:
                    break
                # verified_online.csv has a 'url' column
                if 'url' in row:
                    url = row['url']
                    # Label 0 for Block List (Malicious)
                    data.append([url, 0])
                    count += 1
        
        with open(block_list_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['url', 'label'])
            writer.writerows(data)
        print(f"Generated {block_list_path} with {len(data)} records.")
    except Exception as e:
        print(f"Error processing block list: {e}")

if __name__ == "__main__":
    process_white_list()
    process_block_list()
