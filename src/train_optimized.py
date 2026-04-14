
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pandas as pd
import numpy as np

# Import project modules
from models import MultimodalDrowsinessDetector
from data_loader import DrowsinessDataset, get_transforms

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32 # Increased batch size slightly for better GPU utilization
LEARNING_RATE = 0.0001
EPOCHS = 10
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'data')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, '..', 'best_model.pth')

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, eeg_data, labels in pbar:
        images, eeg_data, labels = images.to(DEVICE), eeg_data.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed Precision Context (Enable only if CUDA)
        use_amp = (DEVICE == 'cuda')
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images, eeg_data)
            loss = criterion(outputs, labels)
            
        # Scaled Backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
        
    return running_loss / len(loader), correct / total

def main():
    print(f"--- FastViT Training Optimization ---")
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("Comparison: FastViT + AMP (Mixed Precision) enabled.")
    
    # Enable Cudnn Benchmark if using CUDA
    if DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Load Data
    train_csv_path = os.path.join(DATA_DIR, 'train_dataset.csv')
    if not os.path.exists(train_csv_path):
        print(f"Error: {train_csv_path} not found.")
        return

    print(f"Loading dataset from {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path)
    
    # Load EEG Features Master File
    eeg_npy_path = os.path.join(DATA_DIR, 'eeg_features.npy')
    if os.path.exists(eeg_npy_path):
        print(f"Loading EEG features from {eeg_npy_path}...")
        eeg_features = np.load(eeg_npy_path)
    else:
        print("Warning: eeg_features.npy not found. EEG data will be zeroed.")
        eeg_features = None
    
    transform = get_transforms()
    train_dataset = DrowsinessDataset(data_frame=train_df, 
                                      transform=transform,
                                      eeg_features=eeg_features,
                                      eeg_length=32)
    
    # Num workers=0 for Windows safety, can increase on Linux
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(DEVICE=='cuda'))
    
    # Initialize FastViT Model
    print("Initializing FastViT-T8 model...")
    model = MultimodalDrowsinessDetector(
        num_classes=2, 
        thermal_model_name='fastvit_t8', # The new speedy model
        eeg_channels=10, 
        eeg_length=32
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Gradient Scaler for AMP (Only for CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Pass scaler/amp_enabled explicit or handle inside train_one_epoch?
        # Let's adjust train_one_epoch too or just handle it globally via the scaler enablement
        pass 
        # Actually I need to update train_one_epoch too to handle cpu case for autocast

        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}, Acc: {acc:.4%}")
        
    total_time = time.time() - start_time
    print(f"Training Complete. Total time: {total_time:.2f}s")
    
    # Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
