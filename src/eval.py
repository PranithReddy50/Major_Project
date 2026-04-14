import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time

# Use exact identical dataset wrapper to ensure it loads exactly as `train.py`
from torch.utils.data import Dataset
from PIL import Image

class MemoryDrowsinessDataset(Dataset):
    def __init__(self, dataframe, eeg_data, transform=None):
        self.df = dataframe
        self.eeg_data = eeg_data
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['thermal_path']
        
        # --- DYNAMIC PATH RECOVERY ---
        import os
        if "thermal.v3-myimages" in img_path:
            norm_path = img_path.replace('\\', '/')
            parts = norm_path.split('thermal.v3-myimages')
            if len(parts) > 1:
                rel_path = 'thermal.v3-myimages' + parts[-1]
                # Assuming we are in src/
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                img_path = os.path.join(project_root, rel_path)
        # -----------------------------
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (256, 256))
            
        if self.transform:
            image = self.transform(image)
            
        eeg_idx = int(row['eeg_index'])
        eeg_signal = self.eeg_data[eeg_idx]
        eeg_tensor = torch.tensor(eeg_signal, dtype=torch.float32)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        
        return image, eeg_tensor, label

def evaluate_model():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
        sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

    from models import MultimodalDrowsinessDetector
    from data_loader import get_transforms

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading Validations on {DEVICE}...")

    # Load data
    val_csv = os.path.join(PROJECT_ROOT, 'data', 'val_dataset.csv')
    eeg_npy = os.path.join(PROJECT_ROOT, 'data', 'eeg_features.npy')
    
    if not os.path.exists(val_csv) or not os.path.exists(eeg_npy):
        print("Dataset not found!")
        return

    val_df = pd.read_csv(val_csv)
    all_eeg_features = np.load(eeg_npy)

    val_ds = MemoryDrowsinessDataset(val_df, all_eeg_features, transform=get_transforms())
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Load Model
    model = MultimodalDrowsinessDetector(
        num_classes=2, 
        thermal_model_name='fastvit_t8',
        eeg_channels=10, 
        eeg_length=32
    ).to(DEVICE)
    
    model_path = os.path.join(PROJECT_ROOT, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ Loaded checkpoint: {model_path}")
    else:
        print("❌ No best_model.pth found! Falling back to untrained weights.")

    model.eval()
    
    all_preds_combined = []
    all_preds_thermal = []
    all_preds_eeg = []
    
    all_labels = []
    total_latency = 0
    
    print("Running Ablation Inference over Validation Set...")
    with torch.no_grad():
        for images, eegs, labels in val_loader:
            images, eegs = images.to(DEVICE), eegs.to(DEVICE)
            
            # --- 1. Combined ---
            start_time = time.time()
            out_c = model(images, eegs)
            total_latency += (time.time() - start_time)
            all_preds_combined.extend(torch.argmax(out_c, dim=1).cpu().numpy())
            
            # --- 2. Thermal Only (Mock EEG with Zeros) ---
            out_t = model(images, torch.zeros_like(eegs).to(DEVICE))
            all_preds_thermal.extend(torch.argmax(out_t, dim=1).cpu().numpy())
            
            # --- 3. EEG Only (Mock Thermal with Zeros) ---
            out_e = model(torch.zeros_like(images).to(DEVICE), eegs)
            all_preds_eeg.extend(torch.argmax(out_e, dim=1).cpu().numpy())
            
            all_labels.extend(labels.numpy())

    # Aggregate Metrics
    avg_latency_ms = (total_latency / len(val_loader.dataset)) * 1000

    # Combined Metrics
    acc_c = accuracy_score(all_labels, all_preds_combined)
    prec_c = precision_score(all_labels, all_preds_combined, average='weighted', zero_division=0)
    rec_c = recall_score(all_labels, all_preds_combined, average='weighted', zero_division=0)
    f1_c = f1_score(all_labels, all_preds_combined, average='weighted', zero_division=0)
    cm_c = confusion_matrix(all_labels, all_preds_combined)
    
    # Ablated Metrics
    acc_t = accuracy_score(all_labels, all_preds_thermal)
    acc_e = accuracy_score(all_labels, all_preds_eeg)
    
    print("\n" + "="*50)
    print("📈 MULTIMODAL ABLATION METRICS")
    print("="*50)
    print(f"1. Combined Accuracy: {acc_c*100:.2f}%")
    print(f"2. Thermal Accuracy:  {acc_t*100:.2f}%")
    print(f"3. EEG Accuracy:      {acc_e*100:.2f}%")
    print("-" * 50)
    print(f"Combined F1-Score:    {f1_c*100:.2f}%")
    print(f"Combined Precision:   {prec_c*100:.2f}%")
    print(f"Combined Recall:      {rec_c*100:.2f}%")
    print(f"Inference Latency:    {avg_latency_ms:.2f} ms per sample")
    
    print("\n📊 CONFUSION MATRIX (Combined)")
    print("-------------")
    print("True \\ Pred | Alert (0) | Drowsy (1)")
    print("------------------------------------")
    print(f"Alert (0)   | {cm_c[0][0]:<9} | {cm_c[0][1]:<9}")
    print(f"Drowsy (1)  | {cm_c[1][0]:<9} | {cm_c[1][1]:<9}")
    print("-------------")
    
    print("\n📝 CLASSIFICATION REPORT")
    print(classification_report(all_labels, all_preds_combined, target_names=['Alert (0)', 'Drowsy (1)'], zero_division=0))
    
    # Save the metrics to a file so app.py can read them
    results_path = os.path.join(PROJECT_ROOT, "eval_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"{acc_c},{acc_t},{acc_e},{prec_c},{rec_c},{f1_c},{cm_c[0][0]},{cm_c[0][1]},{cm_c[1][0]},{cm_c[1][1]}\n")


if __name__ == "__main__":
    evaluate_model()
