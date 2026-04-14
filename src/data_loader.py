import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms

class DrowsinessDataset(Dataset):
    def __init__(self, data_frame, transform=None, eeg_length=256, eeg_features=None):
        """
        Args:
            data_frame (pd.DataFrame): DataFrame with columns ['thermal_path', 'eeg_path', 'label'] or ['thermal_path', 'eeg_index', 'label']
            transform (callable, optional): Transform to be applied on thermal image.
            eeg_length (int): Fixed length to pad/crop EEG signals to.
            eeg_features (np.ndarray, optional): Master array of EEG features if using eeg_index.
        """
        self.data_frame = data_frame
        self.transform = transform
        self.eeg_length = eeg_length
        self.eeg_features = eeg_features

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        
        # Load Thermal Image
        img_path = row['thermal_path']
        
        # --- DYNAMIC PATH RECOVERY ---
        # Automatically fix broken absolute paths from moved project folders
        if "thermal.v3-myimages" in img_path:
            norm_path = img_path.replace('\\', '/')
            parts = norm_path.split('thermal.v3-myimages')
            if len(parts) > 1:
                rel_path = 'thermal.v3-myimages' + parts[-1]
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                img_path = os.path.join(project_root, rel_path)
        # -----------------------------
        
        try:
            image = Image.open(img_path).convert('RGB') # FastViT expects 3 channels
        except Exception as e:
            # Fallback for missing files in development
            image = Image.new('RGB', (256, 256))

        if self.transform:
            image = self.transform(image)

        # Load EEG Signal
        if self.eeg_features is not None and 'eeg_index' in row:
            # Mode A: Load from master array using index
            try:
                idx_ptr = int(row['eeg_index'])
                eeg_signal = self.eeg_features[idx_ptr] # Shape: (Channels, Time) or (Time, Channels)
                
                # Check for correct shape logic if needed, usually we expect (Channels, Time) for Torch
                # If master array is (N, Channels, Time), we are good.
                pass 
            except Exception as e:
                # print(f"Error loading EEG index {idx_ptr}: {e}")
                eeg_signal = np.zeros((4, self.eeg_length), dtype=np.float32)
        else:
            # Mode B: Load from individual files
            try:
                eeg_path = row['eeg_path']
                # Assuming EEG is stored as CSV or Numpy array
                if eeg_path.endswith('.npy'):
                    eeg_signal = np.load(eeg_path)
                elif eeg_path.endswith('.csv'):
                    eeg_df = pd.read_csv(eeg_path)
                    eeg_signal = eeg_df.values.T # Shape: (Channels, Time)
                else:
                    raise ValueError("Unsupported EEG file format")
            except Exception as e:
                eeg_signal = np.zeros((4, self.eeg_length), dtype=np.float32)

        # Preprocessing: Fix length
        eeg_signal = self._process_eeg(eeg_signal) 
        
        # Convert to Tensor
        eeg_tensor = torch.tensor(eeg_signal, dtype=torch.float32)
        label = torch.tensor(int(row['label']), dtype=torch.long)

        return image, eeg_tensor, label

    def _process_eeg(self, signal):
        """
        Ensure signal is (Channels, Fixed_Length)
        """
        # Assume input is (Channels, raw_length)
        if signal.shape[0] > signal.shape[1]:
            # Maybe transposed? Fix if it looks like (Time, Channels)
            if signal.shape[1] <= 32: # Heuristic: Channels usually < Time
                signal = signal.T
        
        channels, length = signal.shape
        target_len = self.eeg_length
        
        if length > target_len:
            # Crop
            start = (length - target_len) // 2
            signal = signal[:, start:start+target_len]
        elif length < target_len:
            # Pad
            padding = np.zeros((channels, target_len - length))
            signal = np.hstack((signal, padding))
            
        return signal

def get_transforms(img_size=256):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def create_dummy_data(rows=10, save_dir='data'):
    """
    Creates dummy data for testing the pipeline when datasets are missing.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    data = []
    for i in range(rows):
        # Create dummy image
        img_path = os.path.join(save_dir, f'img_{i}.png')
        Image.new('RGB', (256, 256)).save(img_path)
        
        # Create dummy EEG
        eeg_path = os.path.join(save_dir, f'eeg_{i}.npy')
        np.save(eeg_path, np.random.randn(4, 300)) # 4 channels, 300 dim
        
        data.append({
            'thermal_path': img_path,
            'eeg_path': eeg_path,
            'label': np.random.randint(0, 2)
        })
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(save_dir, 'dataset.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

if __name__ == "__main__":
    # Test
    csv_file = create_dummy_data(rows=5)
    df = pd.read_csv(csv_file)
    ds = DrowsinessDataset(df, transform=get_transforms())
    img, eeg, lbl = ds[0]
    print(f"Image: {img.shape}, EEG: {eeg.shape}, Label: {lbl}")
