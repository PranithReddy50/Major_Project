import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Force flush if available
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Relative Imports or Add Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR) # Add src to path

try:
    from models import MultimodalDrowsinessDetector
    from data_loader import DrowsinessDataset, get_transforms
except ImportError as e:
    print(f"Import Error: {e}")
    # Try local import if running from src
    try:
        from src.models import MultimodalDrowsinessDetector
        from src.data_loader import DrowsinessDataset, get_transforms
    except:
        print("CRITICAL: Models not found.")
        sys.exit(1)

# Configuration
BATCH_SIZE = 16
EPOCHS = 1 ##########################
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress = tqdm(loader, desc="Training", leave=False)
    for images, eegs, labels in progress:
        images, eegs, labels = images.to(device), eegs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, eegs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, eegs, labels in tqdm(loader, desc="Validation", leave=False):
            images, eegs, labels = images.to(device), eegs.to(device), labels.to(device)

            outputs = model(images, eegs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1

def main():
    print(f"Using device: {DEVICE}")
    print(f"Project Root: {PROJECT_ROOT}")
    
    train_csv = os.path.join(DATA_DIR, 'train_dataset.csv')
    val_csv = os.path.join(DATA_DIR, 'val_dataset.csv')
    eeg_npy = os.path.join(DATA_DIR, 'eeg_features.npy')
    
    if not os.path.exists(train_csv):
        print(f"Dataset not found at {train_csv}")
        return

    print("Loading datasets...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # Load all EEG data into memory to pass to Dataset
    # The Dataset class in data_loader.py expects a path relative to the row? 
    # Wait, my previous data_loader.py implementation anticipated 'eeg_path' in the CSV.
    # But preprocess.py generated 'eeg_index' and saved one big .npy file.
    # I MUST UPDATE data_loader.py to handle this 'Index Look-up' strategy or update train.py to pass the big array.
    
    # Let's load the big array here and modify Dataset to accept it.
    all_eeg_features = np.load(eeg_npy)
    print(f"Loaded EEG Features: {all_eeg_features.shape}")
    
    # Update Dataset to use In-Memory EEG
    # We need to monkey-patch or update the class. Let's update the class instance logic here.
    # Actually, simpler to just update the 'eeg_path' in the dataframe to be the specific row? 
    # No, that's inefficient. 
    # Better: Update `data_loader.py` to support `eeg_data_array` argument.
    
    # For now, let's just make a custom Dataset wrapper here to avoid modifying too many files and breaking imports.
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
            
            # Image
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
                
            # EEG
            eeg_idx = int(row['eeg_index'])
            eeg_signal = self.eeg_data[eeg_idx] # Shape (Channels, Time) or (Time, Channels) depending on preprocess
            # Preprocess.py saved (Feat, Time) -> (10, 32)
            
            eeg_tensor = torch.tensor(eeg_signal, dtype=torch.float32)
            label = torch.tensor(int(row['label']), dtype=torch.long)
            
            return image, eeg_tensor, label

    train_ds = MemoryDrowsinessDataset(train_df, all_eeg_features, transform=get_transforms())
    val_ds = MemoryDrowsinessDataset(val_df, all_eeg_features, transform=get_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing Model...")
    # FastViT + EEG (10 channels, 32 length)
    model = MultimodalDrowsinessDetector(
        num_classes=2, 
        thermal_model_name='fastvit_t8',
        eeg_channels=10, 
        eeg_length=32
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, 'best_model.pth'))
            print("Saved Best Model!")

if __name__ == "__main__":
    main()
