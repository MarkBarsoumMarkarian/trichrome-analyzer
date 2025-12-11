"""
Advanced Features for Trichrome Analyzer
=========================================

1. Deep Learning Segmentation (U-Net)
2. Ensemble Models
3. Active Learning Pipeline

Requirements:
    pip install torch torchvision

Usage:
    # Train U-Net
    python trichrome_advanced.py train-unet --images data/images --masks data/masks
    
    # Create ensemble
    python trichrome_advanced.py ensemble --traditional model.pkl --unet unet.pth --val validation.csv
    
    # Active learning
    python trichrome_advanced.py active-learn --ensemble ensemble.pkl --unlabeled data/unlabeled
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# ==================== U-NET MODEL ====================

class UNet(nn.Module):
    """
    U-Net for semantic segmentation
    
    Input: RGB image (3 channels)
    Output: Segmentation mask (3 classes: background, fibrosis, glomeruli)
    """
    
    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        super(UNet, self).__init__()
        
        features = init_features
        
        # Encoder
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 
                                          kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 
                                          kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 
                                          kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 
                                          kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)
        
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class TrichromeDataset(Dataset):
    """Dataset for U-Net training"""
    
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask


def train_unet(train_loader, val_loader, num_epochs=50, device='cuda'):
    """Train U-Net model"""
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5
    )
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'unet_best.pth')
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")
    
    return model, history


def predict_with_unet(model, image, device='cuda'):
    """Predict with U-Net"""
    original_shape = image.shape[:2]
    image_resized = cv2.resize(image, (256, 256))
    image_tensor = torch.from_numpy(
        image_resized.transpose(2, 0, 1)
    ).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    prediction = cv2.resize(
        prediction, (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    return prediction


# ==================== ENSEMBLE ====================

class EnsemblePredictor:
    """Ensemble of multiple models"""
    
    def __init__(self):
        self.models = []
        self.weights = None
        
    def add_model(self, model, model_type, device='cpu'):
        """Add model to ensemble"""
        self.models.append({
            'model': model,
            'type': model_type,
            'device': device
        })
    
    def fit_weights(self, X_val, y_val):
        """Learn optimal ensemble weights"""
        from scipy.optimize import minimize
        
        predictions = []
        for model_dict in self.models:
            preds = []
            for x in X_val:
                pred = self._predict_single(model_dict, x)
                preds.append(pred)
            predictions.append(preds)
        
        predictions = np.array(predictions).T
        
        def ensemble_error(weights):
            weights = weights / weights.sum()
            ensemble_pred = predictions @ weights
            return np.mean((ensemble_pred - y_val) ** 2)
        
        w0 = np.ones(len(self.models)) / len(self.models)
        
        result = minimize(
            ensemble_error, w0,
            constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
            bounds=[(0, 1)] * len(self.models)
        )
        
        self.weights = result.x
        print(f"Optimal weights: {self.weights}")
    
    def predict(self, image):
        """Predict with ensemble"""
        if self.weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        predictions = []
        for model_dict in self.models:
            pred = self._predict_single(model_dict, image)
            predictions.append(pred)
        
        ensemble_pred = np.average(predictions, weights=self.weights)
        
        return ensemble_pred, predictions
    
    def _predict_single(self, model_dict, image):
        """Predict with single model"""
        if model_dict['type'] == 'traditional':
            analyzer = model_dict['model']
            _, ml_pred, _ = analyzer.predict(image)
            return ml_pred
        
        elif model_dict['type'] == 'unet':
            unet_model = model_dict['model']
            device = model_dict['device']
            mask = predict_with_unet(unet_model, image, device)
            fibrosis_pct = (np.sum(mask == 1) / mask.size) * 100
            return fibrosis_pct
        
        else:
            raise ValueError(f"Unknown model type: {model_dict['type']}")
    
    def save(self, filepath):
        """Save ensemble config"""
        data = {
            'weights': self.weights.tolist() if self.weights is not None else None,
            'model_types': [m['type'] for m in self.models],
            'n_models': len(self.models)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Ensemble config saved to {filepath}")


# ==================== ACTIVE LEARNING ====================

class ActiveLearningPipeline:
    """Active learning for efficient model improvement"""
    
    def __init__(self, model, uncertainty_threshold=0.15):
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertain_samples = []
        self.labeled_pool = []
        self.unlabeled_pool = []
    
    def add_unlabeled_data(self, image_paths):
        """Add unlabeled images"""
        self.unlabeled_pool.extend(image_paths)
    
    def select_samples(self, n_samples=10, strategy='uncertainty'):
        """
        Select most informative samples
        
        Args:
            n_samples: Number to select
            strategy: 'uncertainty', 'diversity', or 'hybrid'
        """
        if strategy == 'uncertainty':
            return self._uncertainty_sampling(n_samples)
        elif strategy == 'diversity':
            return self._diversity_sampling(n_samples)
        elif strategy == 'hybrid':
            return self._hybrid_sampling(n_samples)
    
    def _uncertainty_sampling(self, n_samples):
        """Select samples with highest uncertainty"""
        uncertainties = []
        
        for img_path in self.unlabeled_pool:
            image = cv2.imread(img_path)
            
            if hasattr(self.model, 'predict'):
                ensemble_pred, individual_preds = self.model.predict(image)
                uncertainty = np.std(individual_preds)
            else:
                uncertainty = 0.1
            
            uncertainties.append((img_path, uncertainty))
        
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        self.uncertain_samples = uncertainties[:n_samples]
        
        return [path for path, unc in uncertainties[:n_samples]]
    
    def _diversity_sampling(self, n_samples):
        """Select diverse samples"""
        from sklearn.cluster import KMeans
        
        features_list = []
        
        for img_path in self.unlabeled_pool:
            image = cv2.imread(img_path)
            features = self._extract_diversity_features(image)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(features_array)
        
        selected_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(features_array - center, axis=1)
            selected_indices.append(np.argmin(distances))
        
        return [self.unlabeled_pool[i] for i in selected_indices]
    
    def _hybrid_sampling(self, n_samples):
        """Combine uncertainty and diversity"""
        n_uncertain = n_samples // 2
        n_diverse = n_samples - n_uncertain
        
        uncertain = self._uncertainty_sampling(n_uncertain)
        
        remaining = [p for p in self.unlabeled_pool if p not in uncertain]
        temp_pool = self.unlabeled_pool
        self.unlabeled_pool = remaining
        
        diverse = self._diversity_sampling(n_diverse)
        
        self.unlabeled_pool = temp_pool
        
        return uncertain + diverse
    
    def _extract_diversity_features(self, image):
        """Extract features for diversity"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        features = np.concatenate([
            hist_h.flatten(), hist_s.flatten(), hist_v.flatten()
        ])
        features = features / features.sum()
        
        return features
    
    def generate_report(self, output_path='active_learning_report.png'):
        """Generate visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if self.uncertain_samples:
            uncertainties = [unc for _, unc in self.uncertain_samples]
            axes[0].hist(uncertainties, bins=20, edgecolor='black', alpha=0.7)
            axes[0].axvline(self.uncertainty_threshold, color='red', 
                           linestyle='--', label=f'Threshold: {self.uncertainty_threshold}')
            axes[0].set_xlabel('Prediction Uncertainty')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Uncertainty Distribution')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
        
        pool_sizes = [
            len(self.labeled_pool),
            len(self.unlabeled_pool),
            len(self.uncertain_samples)
        ]
        labels = ['Labeled', 'Unlabeled', 'High Uncertainty']
        colors = ['#48bb78', '#4299e1', '#f56565']
        
        axes[1].bar(labels, pool_sizes, color=colors, edgecolor='black')
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title('Active Learning Data Pools')
        axes[1].grid(axis='y', alpha=0.3)
        
        for i, (label, size) in enumerate(zip(labels, pool_sizes)):
            axes[1].text(i, size + max(pool_sizes)*0.02, str(size),
                        ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Report saved to {output_path}")


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    import sys
    from sklearn.model_selection import train_test_split
    
    parser = argparse.ArgumentParser(description='Advanced Trichrome Features')
    parser.add_argument('command', 
                       choices=['train-unet', 'ensemble', 'active-learn'],
                       help='Command to execute')
    parser.add_argument('--images', type=str, help='Image directory')
    parser.add_argument('--masks', type=str, help='Mask directory')
    parser.add_argument('--traditional', type=str, help='Traditional model path')
    parser.add_argument('--unet', type=str, help='U-Net model path')
    parser.add_argument('--val', type=str, help='Validation CSV')
    parser.add_argument('--unlabeled', type=str, help='Unlabeled images folder')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--output', type=str, help='Output file')
    
    args = parser.parse_args()
    
    if args.command == 'train-unet':
        print("Training U-Net...")
        print(f"Looking for images in: {Path(args.images).absolute()}")
        print(f"Looking for masks in: {Path(args.masks).absolute()}")
        
        # Find all images (multiple extensions)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        all_image_files = []
        for ext in image_extensions:
            all_image_files.extend(Path(args.images).rglob(ext))
        
        print(f"\nFound {len(all_image_files)} total images")
        
        # Find matching mask files
        image_paths = []
        mask_paths = []
        
        for img_path in sorted(all_image_files):
            # Try multiple mask naming conventions
            possible_masks = [
                # Same structure, .png extension
                Path(args.masks) / img_path.relative_to(args.images).with_suffix('.png'),
                # Same structure, _mask.png suffix
                Path(args.masks) / (img_path.stem + '_mask.png'),
                # Flat structure in masks folder
                Path(args.masks) / img_path.with_suffix('.png').name,
                Path(args.masks) / (img_path.stem + '_mask.png'),
            ]
            
            for mask_path in possible_masks:
                if mask_path.exists():
                    image_paths.append(str(img_path))
                    mask_paths.append(str(mask_path))
                    break
        
        print(f"Found {len(image_paths)} matching image-mask pairs")
        
        if len(image_paths) == 0:
            print("\n❌ ERROR: No matching image-mask pairs found!")
            print("\nExpected structure:")
            print("Option 1 (mirrored structure):")
            print("  images/")
            print("    ├── img1.jpg")
            print("    └── img2.jpg")
            print("  masks/")
            print("    ├── img1.png")
            print("    └── img2.png")
            print("\nOption 2 (flat with _mask suffix):")
            print("  images/")
            print("    ├── img1.jpg")
            print("  masks/")
            print("    ├── img1_mask.png")
            
            # Show what we actually found
            if len(all_image_files) > 0:
                print(f"\nImages found:")
                for i, img in enumerate(sorted(all_image_files)[:5]):
                    print(f"  - {img.relative_to(args.images)}")
                if len(all_image_files) > 5:
                    print(f"  ... and {len(all_image_files)-5} more")
                
                print("\nMasks directory contents:")
                mask_dir = Path(args.masks)
                if mask_dir.exists():
                    mask_files_found = list(mask_dir.rglob('*.png'))
                    if mask_files_found:
                        for i, mask in enumerate(sorted(mask_files_found)[:5]):
                            print(f"  - {mask.name}")
                        if len(mask_files_found) > 5:
                            print(f"  ... and {len(mask_files_found)-5} more")
                    else:
                        print("  (no .png files found)")
                else:
                    print(f"  Directory does not exist!")
            
            sys.exit(1)
        
        print("\nExample pairs:")
        for i in range(min(3, len(image_paths))):
            print(f"  {Path(image_paths[i]).name} -> {Path(mask_paths[i]).name}")
        
        # Split train/val
        if len(image_paths) < 5:
            print(f"\n⚠️  WARNING: Only {len(image_paths)} pairs found. Need at least 5 for training.")
            print("Continuing anyway, but results may be poor...")
        
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            image_paths, mask_paths, test_size=0.2, random_state=42
        )
        
        train_dataset = TrichromeDataset(train_imgs, train_masks, augment=True)
        val_dataset = TrichromeDataset(val_imgs, val_masks, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, history = train_unet(train_loader, val_loader, args.epochs, device)
        
        print("✓ Training complete! Model saved to unet_best.pth")
    
    elif args.command == 'ensemble':
        print("Creating ensemble...")
        
        # Load models
        from trichrome_core import TrichromeAnalyzer
        import pandas as pd
        
        ensemble = EnsemblePredictor()
        
        # Traditional model
        traditional = TrichromeAnalyzer()
        traditional.load_model(args.traditional)
        ensemble.add_model(traditional, 'traditional')
        
        # U-Net
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        unet = UNet()
        unet.load_state_dict(torch.load(args.unet, map_location=device))
        unet.to(device)
        ensemble.add_model(unet, 'unet', device)
        
        # Fit weights
        df = pd.read_csv(args.val)
        X_val = [cv2.imread(p) for p in df['filepath']]
        y_val = df['ground_truth'].values
        
        ensemble.fit_weights(X_val, y_val)
        
        # Save
        output = args.output or 'ensemble_config.json'
        ensemble.save(output)
    
    elif args.command == 'active-learn':
        print("Running active learning...")
        
        # Load ensemble
        with open(args.ensemble, 'rb') as f:
            ensemble = pickle.load(f)
        
        active_learner = ActiveLearningPipeline(ensemble)
        
        # Add unlabeled images
        unlabeled_files = list(Path(args.unlabeled).rglob('*.jpg'))
        active_learner.add_unlabeled_data([str(p) for p in unlabeled_files])
        
        # Select samples
        selected = active_learner.select_samples(n_samples=10, strategy='hybrid')
        
        print("\n=== Selected for Labeling ===")
        for i, path in enumerate(selected, 1):
            print(f"{i}. {Path(path).name}")
        
        # Save list
        output = args.output or 'samples_to_label.txt'
        with open(output, 'w') as f:
            for path in selected:
                f.write(f"{path}\n")
        
        active_learner.generate_report()
        print(f"\n✓ Sample list saved to {output}")