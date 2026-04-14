import torch
import numpy as np
from PIL import Image
import os
import sys
import pandas as pd
import cv2

# Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

from models import MultimodalDrowsinessDetector
from data_loader import get_transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.path.join(PROJECT_ROOT, 'best_model.pth')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def load_trained_model(device=DEVICE):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None

    # Re-instantiate model with same params as training
    # UPGRADE: Switched to FastViT-T8 (Apple, 2023) for 3x speedup
    model = MultimodalDrowsinessDetector(
        num_classes=2, 
        thermal_model_name='fastvit_t8', 
        eeg_channels=10, 
        eeg_length=32
    ).to(device)
    
    # Load state dict
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
         print(f"Error loading state dict: {e}")
         return None

    model.eval()
    
    # OPTIMIZATION: Compile model - DISABLED for Windows compatibility (requires VS Build Tools)
    # if hasattr(torch, 'compile'):
    #     print("Optimizing model with torch.compile()...")
    #     try:
    #         model = torch.compile(model)
    #     except Exception as e:
    #         print(f"Compilation skipped: {e}")

    print("Model loaded successfully.")
    return model

def predict_single(model, image, eeg_data):
    """
    Predicts drowsiness for a single thermal image and EEG segment.
    
    Args:
        model: Loaded PyTorch model
        image: PIL Image
        eeg_data: numpy array or list of shape (10, 32)
        
    Returns:
        dict: {
            'label': str ('Drowsy' or 'Alert'), 
            'confidence': float,
            'probs': list
        }
    """
    transform = get_transforms()
    # Ensure image is RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Process EEG
    # If eeg_data is a flat list/string, try to shape it, otherwise expect valid shape
    if isinstance(eeg_data, (list, np.ndarray)):
        eeg_arr = np.array(eeg_data, dtype=np.float32)
        if eeg_arr.size != 10 * 32:
             # Fallback or error - for now padding or truncating could be dangerous, 
             # let's assume the UI handles valid 10x32 input or we handle it here.
             # For robustness, if shape is wrong, we might need to mock or error.
             pass 
             
        # Reshape if flat
        if eeg_arr.ndim == 1 and eeg_arr.size == 320:
            eeg_arr = eeg_arr.reshape(10, 32)
        elif eeg_arr.ndim == 1:
             # Just mock if length is totally off (demo purposes)
             # print("Warning: EEG data shape incorrect, using zeroes")
             # eeg_arr = np.zeros((10, 32), dtype=np.float32)
             pass 
    
    eeg_tensor = torch.tensor(eeg_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_tensor, eeg_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    label = "Drowsy" if pred_idx == 1 else "Alert"
    confidence = probs[0][pred_idx].item()
    
    return {
        'label': label,
        'confidence': confidence,
        'probs': probs[0].tolist()
    }

import matplotlib.pyplot as plt

def get_best_rotation(pil_image):
    """
    Rotates the image (0, 90, 180, 270) to find the one where a face is detected best.
    Returns the rotated PIL image.
    """
    try:
        # Convert PIL to CV2 BGR
        img_np = np.array(pil_image.convert('RGB')) 
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Safe Enhance
        try:
            img_gray_eq = cv2.equalizeHist(img_gray_orig)
        except:
            img_gray_eq = img_gray_orig
            
        # Load Cascades - Try standard cv2 path or skip
        cascades = []
        try:
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                p = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                if os.path.exists(p):
                    cascades.append(cv2.CascadeClassifier(p))
                    
                p2 = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml')
                if os.path.exists(p2):
                    cascades.append(cv2.CascadeClassifier(p2))
        except:
            pass
                
        if not cascades:
            return pil_image
            
        best_rotation = 0
        max_score = 0
        found_face = False
        
        rotations = [0, 90, 180, 270]
        
        for angle in rotations:
            try:
                # Rotate gray image
                if angle == 0:
                    r_gray = img_gray_eq
                elif angle == 90:
                    r_gray = cv2.rotate(img_gray_eq, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    r_gray = cv2.rotate(img_gray_eq, cv2.ROTATE_180)
                elif angle == 270:
                    r_gray = cv2.rotate(img_gray_eq, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Detect
                for cascade in cascades:
                    faces = cascade.detectMultiScale(r_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        area = max(w*h for (x,y,w,h) in faces)
                        if area > max_score:
                            max_score = area
                            best_rotation = angle
                            found_face = True
            except:
                continue
        
        if found_face:
            if best_rotation == 0:
                return pil_image
            elif best_rotation == 90:
                return pil_image.rotate(-90, expand=True) # 90 CW
            elif best_rotation == 180:
                return pil_image.rotate(180, expand=True)
            elif best_rotation == 270:
                return pil_image.rotate(90, expand=True) # 270 CW = 90 CCW
                
    except Exception as e:
        # Fail gracefully
        pass
        
    return pil_image

def predict_batch(model, num_samples=20):
    val_csv = os.path.join(DATA_DIR, 'val_dataset.csv')
    eeg_npy = os.path.join(DATA_DIR, 'eeg_features.npy')
    vis_dir = os.path.join(PROJECT_ROOT, 'inference_results')
    metrics_file = os.path.join(PROJECT_ROOT, 'metrics_output.txt')
    
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    output_lines = []
    
    if not os.path.exists(val_csv):
        print("Validation data not found.")
        return

    df = pd.read_csv(val_csv)
    all_eeg = np.load(eeg_npy)
    
    output_lines.append("\n" + "="*80)
    output_lines.append(f"{'ID':<5} | {'Image File':<30} | {'True':<10} | {'Pred':<10} | {'Conf':<6}")
    output_lines.append("="*80)
    
    indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
    
    # Metrics counters
    tp = 0 # True Pos (Drowsy=1, Pred=1)
    tn = 0 # True Neg (Alert=0, Pred=0)
    fp = 0 # False Pos (Alert=0, Pred=1)
    fn = 0 # False Neg (Drowsy=1, Pred=0)
    
    correct_count = 0
    
    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        
        # Image
        img_path = row['thermal_path']
        fname = os.path.basename(img_path)
        try:
            image_raw = Image.open(img_path).convert('RGB')
            # Apply Rotation Logic Here
            image = get_best_rotation(image_raw)
        except Exception as e:
            # print(f"Error loading {fname}: {e}")
            image = Image.new('RGB', (256, 256))
            
        transform = get_transforms()
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # EEG
        eeg_idx = int(row['eeg_index'])
        eeg_data = all_eeg[eeg_idx] # (Channels, Time)
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor, eeg_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
        true_label_idx = int(row['label'])
        true_label = "Drowsy" if true_label_idx == 1 else "Alert"
        pred_label = "Drowsy" if pred_idx == 1 else "Alert"
        confidence = probs[0][pred_idx].item()
        
        # Metrics update
        if true_label_idx == 1:
            if pred_idx == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred_idx == 0:
                tn += 1
            else:
                fp += 1
                
        if true_label_idx == pred_idx:
            correct_count += 1
        
        # Color coding (conceptual for CLI)
        match = "✅" if true_label == pred_label else "❌"
        
        line = f"{i+1:<5} | {fname:<30.28} | {true_label:<10} | {pred_label:<10} | {confidence:.2f} {match}"
        print(line)
        output_lines.append(line)
        
        # --- Visualization ---
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            # Thermal Image
            axes[0].imshow(image)
            axes[0].set_title(f"Thermal Input\n{fname}")
            axes[0].axis('off')
            
            # EEG Signal (Plot first 3 channels for clarity)
            axes[1].plot(eeg_data[0], label='Attention') # Approx labels based on index
            axes[1].plot(eeg_data[2], label='Delta')
            axes[1].plot(eeg_data[3], label='Theta')
            axes[1].set_title(f"EEG Signal (Segment {eeg_idx})\nTrue: {true_label}")
            axes[1].legend(loc='upper right', fontsize='small')
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle(f"Prediction: {pred_label} ({confidence:.1%})", fontsize=14, color='green' if match=="✅" else 'red')
            plt.tight_layout()
            
            # Save
            save_path = os.path.join(vis_dir, f"result_{i+1}_{pred_label}.png")
            plt.savefig(save_path)
            plt.close()
        except:
            pass

    total = tp + tn + fp + fn
    accuracy = correct_count / total if total > 0 else 0.0
    
    output_lines.append("="*80 + "\n")
    output_lines.append("")
    output_lines.append("="*40)
    output_lines.append("METRICS REPORT")
    output_lines.append("="*40)
    output_lines.append(f"Total Samples  : {total}")
    output_lines.append(f"Accuracy       : {accuracy:.2%}")
    output_lines.append("-" * 20)
    output_lines.append(f"Confusion Matrix:")
    output_lines.append(f"{'':>10} {'Pred Alert':>12} {'Pred Drowsy':>12}")
    output_lines.append(f"{'True Alert':>10} {tn:>12} {fp:>12}")
    output_lines.append(f"{'True Drowsy':>10} {fn:>12} {tp:>12}")
    output_lines.append("="*40 + "\n")
    output_lines.append(f"Visualizations saved to: {vis_dir}")
    
    # Write to file
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
    print(f"Metrics written to {metrics_file}")

if __name__ == "__main__":
    print("Running Inference Batch Test...")
    model = load_trained_model()
    if model:
        predict_batch(model, 20)
