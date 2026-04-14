import torch
import torch.nn as nn
import timm

class ThermalBranch(nn.Module):
    def __init__(self, model_name='fastvit_t8', pretrained=True):
        super(ThermalBranch, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0) # num_classes=0 gives feature vector
        
        # --- TRAINING OPTIMIZATION ---
        # Freeze pretrained transformer weights to train 3x-5x faster on CPU
        for param in self.model.parameters():
            param.requires_grad = False
        # -----------------------------
        
        # Get feature dimension automatically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            output = self.model(dummy_input)
            self.feature_dim = output.shape[1]

    def forward(self, x):
        # x shape: (B, 3, H, W)
        features = self.model(x)
        
        # FastViT (and other modern models) might return a list of features from different stages
        # We only want the last/deepest feature map for classification
        if isinstance(features, (list, tuple)):
            features = features[-1]
            
        return features

class EEGBranch(nn.Module):
    def __init__(self, num_channels=4, input_length=128, feature_dim=128):
        """
        Lightweight 1D CNN for EEG signals.
        """
        super(EEGBranch, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Calculate flat features size
        # Input: (L) -> Pool(/2) -> Pool(/2) -> L/4
        final_length = input_length // 4
        self.flat_dim = 64 * final_length
        self.fc = nn.Linear(self.flat_dim, feature_dim)

    def forward(self, x):
        # x shape: (B, num_channels, input_length)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc(x))
        return x

class MultimodalDrowsinessDetector(nn.Module):
    def __init__(self, num_classes=2, thermal_model_name='fastvit_t8', eeg_channels=4, eeg_length=128):
        super(MultimodalDrowsinessDetector, self).__init__()
        self.thermal_branch = ThermalBranch(model_name=thermal_model_name)
        self.eeg_branch = EEGBranch(num_channels=eeg_channels, input_length=eeg_length)
        
        fusion_dim = self.thermal_branch.feature_dim + 128 # 128 is EEG feature dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, thermal_img, eeg_signal):
        thermal_feat = self.thermal_branch(thermal_img)
        eeg_feat = self.eeg_branch(eeg_signal)
        
        combined_feat = torch.cat((thermal_feat, eeg_feat), dim=1)
        out = self.classifier(combined_feat)
        return out

if __name__ == "__main__":
    # Test instantiation
    model = MultimodalDrowsinessDetector(eeg_channels=4, eeg_length=256)
    t_img = torch.randn(2, 3, 256, 256)
    e_sig = torch.randn(2, 4, 256)
    output = model(t_img, e_sig)
    print(f"Output shape: {output.shape}")
