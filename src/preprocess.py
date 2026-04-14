import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Force stdout flushing
sys.stdout.reconfigure(line_buffering=True)

try:
    from natsort import natsorted
    print("Imported natsort successfully")
except ImportError:
    print("natsort not found, falling back to sorted")
    natsorted = sorted

# Determine script directory to make paths relative
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Configuration
EEG_FILE = os.path.join(PROJECT_ROOT, 'EEG_Signals_acquiredDataset.xlsx')
THERMAL_DIR = os.path.join(PROJECT_ROOT, 'thermal.v3-myimages-tfwtrainingset.yolov11', 'train', 'images')
SEQ_LENGTH = 32
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')

def main():
    print("Starting preprocessing...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Reading EEG from {EEG_FILE}")
    try:
        # Check if file exists
        if not os.path.exists(EEG_FILE):
             # Try absolute path based on CWD
             cwd = os.getcwd()
             EEG_FILE_ABS = os.path.join(cwd, 'EEG_Signals_acquiredDataset.xlsx')
             if os.path.exists(EEG_FILE_ABS):
                 print(f"Found file at {EEG_FILE_ABS}")
                 df = pd.read_excel(EEG_FILE_ABS)
             else:
                 print(f"ERROR: EEG file not found at {EEG_FILE} or {EEG_FILE_ABS}")
                 return
        else:
            df = pd.read_excel(EEG_FILE)
            
        print("EEG File loaded.")
        print(f"Columns: {df.columns}")
        
    except Exception as e:
        print(f"Error loading EEG: {e}")
        return

    # Process EEG
    feature_cols = [c for c in df.columns if c != 'classification']
    features = df[feature_cols].values
    labels = df['classification'].values
    
    # Normalize
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
    
    X = []
    y = []
    
    for i in range(0, len(features) - SEQ_LENGTH, 1):
        window = features[i:i+SEQ_LENGTH]
        label = labels[i+SEQ_LENGTH-1]
        X.append(window.T)
        y.append(label)
        
    X_eeg = np.array(X, dtype=np.float32)
    y_eeg = np.array(y, dtype=np.int64)
    print(f"EEG Processed: {X_eeg.shape}")

    # Process Thermal
    print(f"Scanning images in {THERMAL_DIR}")
    thermal_imgs = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        thermal_imgs.extend(glob.glob(os.path.join(THERMAL_DIR, ext)))
    
    thermal_imgs = natsorted(thermal_imgs)
    print(f"Found {len(thermal_imgs)} images")
    
    if len(thermal_imgs) == 0:
        print("No images found. Exiting.")
        return

    # Alignment
    min_len = min(len(X_eeg), len(thermal_imgs))
    print(f"Alignment limit: {min_len}")
    
    if min_len == 0:
        print("Dataset size is 0. Exiting.")
        return

    X_eeg = X_eeg[:min_len]
    y_eeg = y_eeg[:min_len]
    thermal_imgs = thermal_imgs[:min_len]

    np.save(os.path.join(OUTPUT_DIR, 'eeg_features.npy'), X_eeg)
    
    data_rows = []
    for i in range(min_len):
        data_rows.append({
            'eeg_index': i,
            'thermal_path': os.path.abspath(thermal_imgs[i]),
            'label': y_eeg[i]
        })
        
    df_out = pd.DataFrame(data_rows)
    train_df, val_df = train_test_split(df_out, test_size=0.2, random_state=42) # Removed stratify in case of class imbalance/small size errors
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_dataset.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val_dataset.csv'), index=False)
    
    print("SUCCESS: Preprocessing done.")

if __name__ == "__main__":
    main()
