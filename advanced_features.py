"""
Advanced Features for Trichrome Analyzer
=========================================

1. Color Normalization (Macenko Method)
2. Deep Learning Segmentation (U-Net)
3. Ensemble Models
4. Active Learning Pipeline

Requirements:
    pip install torch torchvision pillow scikit-image spams
"""

import cv2
import numpy as np
from sklearn.decomposition import NMF
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt

# ==================== 1. COLOR NORMALIZATION (MACENKO METHOD) ====================

class MacenkoNormalizer:
    """
    Stain normalization using Macenko method
    Separates H&E stains and normalizes to reference image
    """
    
    def __init__(self):
        self.target_stains = None
        self.target_concentrations = None
        self.maxC_target = None
        
    def fit(self, target_image):
        """
        Fit normalizer to target/reference image
        
        Args:
            target_image: RGB image (H x W x 3), values 0-255
        """
        # Convert to optical density
        target_od = self._rgb_to_od(target_image)
        
        # Remove transparent pixels
        od_hat = target_od[~self._is_transparent(target_od)]
        
        # Compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
        
        # Project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        proj = od_hat @ eigvecs[:, 1:3]
        
        # Find the min and max vectors
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        min_phi = np.percentile(phi, 1)
        max_phi = np.percentile(phi, 99)
        
        v1 = eigvecs[:, 1:3] @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = eigvecs[:, 1:3] @ np.array([np.cos(max_phi), np.sin(max_phi)])
        
        # Make sure vector corresponding to hematoxylin is first
        if v1[0] > v2[0]:
            self.target_stains = np.array([v1, v2]).T
        else:
            self.target_stains = np.array([v2, v1]).T
        
        # Compute concentrations
        self.target_concentrations = self._get_concentrations(target_od, self.target_stains)
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0)
        
        return self
    
    def transform(self, image):
        """
        Normalize image to match target staining
        
        Args:
            image: RGB image to normalize
            
        Returns:
            Normalized RGB image
        """
        if self.target_stains is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Convert to OD
        od = self._rgb_to_od(image)
        
        # Remove transparent pixels
        od_hat = od[~self._is_transparent(od)]
        
        # Compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
        
        # Project
        proj = od_hat @ eigvecs[:, 1:3]
        
        # Find angles
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        min_phi = np.percentile(phi, 1)
        max_phi = np.percentile(phi, 99)
        
        v1 = eigvecs[:, 1:3] @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = eigvecs[:, 1:3] @ np.array([np.cos(max_phi), np.sin(max_phi)])
        
        # Stain matrix
        if v1[0] > v2[0]:
            stains = np.array([v1, v2]).T
        else:
            stains = np.array([v2, v1]).T
        
        # Get concentrations
        concentrations = self._get_concentrations(od, stains)
        
        # Normalize concentrations
        maxC = np.percentile(concentrations, 99, axis=0)
        concentrations *= (self.maxC_target / maxC)
        
        # Recreate image with target stain matrix
        normalized_od = concentrations @ self.target_stains.T
        
        # Convert back to RGB
        normalized = self._od_to_rgb(normalized_od)
        
        return normalized.astype(np.uint8)
    
    def _rgb_to_od(self, image):
        """Convert RGB to optical density"""
        image = image.astype(np.float64)
        image = np.maximum(image, 1)  # Avoid log(0)
        return -np.log(image / 255.0)
    
    def _od_to_rgb(self, od):
        """Convert optical density to RGB"""
        rgb = np.exp(-od) * 255
        return np.clip(rgb, 0, 255)
    
    def _is_transparent(self, od, threshold=0.15):
        """Check if pixel is transparent (background)"""
        return np.all(od < threshold, axis=1)
    
    def _get_concentrations(self, od, stains):
        """Get stain concentrations"""
        od_flat = od.reshape(-1, 3)
        concentrations = np.linalg.lstsq(stains, od_flat.T, rcond=None)[0].T
        return concentrations.reshape(od.shape[:2] + (2,))
    
    def save(self, filepath):
        """Save fitted normalizer"""
        data = {
            'target_stains': self.target_stains,
            'target_concentrations': self.target_concentrations,
            'maxC_target': self.maxC_target
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """Load fitted normalizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.target_stains = data['target_stains']
        self.target_concentrations = data['target_concentrations']
        self.maxC_target = data['maxC_target']
        return self


# ==================== 2. DEEP LEARNING SEGMENTATION (U-NET) ====================

class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation
    
    Input: RGB image (3 channels)
    Output: Segmentation mask (3 classes: background, fibrosis, glomeruli)
    """
    
    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        super(UNet, self).__init__()
        
        features = init_features
        
        # Encoder (downsampling)
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
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)
        
        # Output
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
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
        """Convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class TrichromeDataset(Dataset):
    """Dataset for training U-Net on trichrome images"""
    
    def __init__(self, image_paths, mask_paths, transform=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Resize to fixed size
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Data augmentation
        if self.augment:
            # Random flip
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            
            # Random rotation
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask


def train_unet(train_loader, val_loader, num_epochs=50, device='cuda'):
    """
    Train U-Net model
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
    """
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training
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
        
        # Validation
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
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'unet_best.pth')
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model, history


def predict_with_unet(model, image, device='cuda'):
    """
    Predict segmentation mask using trained U-Net
    
    Args:
        model: Trained U-Net model
        image: Input RGB image (numpy array)
        device: 'cuda' or 'cpu'
    
    Returns:
        Segmentation mask (0=background, 1=fibrosis, 2=glomeruli)
    """
    # Preprocess
    original_shape = image.shape[:2]
    image_resized = cv2.resize(image, (256, 256))
    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Resize back to original
    prediction = cv2.resize(prediction, (original_shape[1], original_shape[0]), 
                           interpolation=cv2.INTER_NEAREST)
    
    return prediction


# ==================== 3. ENSEMBLE MODELS ====================

class EnsemblePredictor:
    """
    Ensemble of multiple models for robust prediction
    
    Combines:
    1. Traditional segmentation + Ridge regression
    2. Traditional segmentation + Random Forest
    3. U-Net segmentation + Ridge regression
    """
    
    def __init__(self):
        self.models = []
        self.weights = None
        self.normalizer = None
        
    def add_model(self, model, model_type='traditional'):
        """
        Add a model to the ensemble
        
        Args:
            model: Trained model (sklearn or pytorch)
            model_type: 'traditional', 'unet', or 'hybrid'
        """
        self.models.append({
            'model': model,
            'type': model_type
        })
    
    def fit_weights(self, X_val, y_val):
        """
        Learn optimal weights for ensemble using validation set
        
        Args:
            X_val: Validation images
            y_val: Ground truth labels
        """
        # Get predictions from each model
        predictions = []
        for model_dict in self.models:
            preds = []
            for x in X_val:
                pred = self._predict_single(model_dict, x)
                preds.append(pred)
            predictions.append(preds)
        
        predictions = np.array(predictions).T  # Shape: (n_samples, n_models)
        
        # Optimize weights to minimize error
        from scipy.optimize import minimize
        
        def ensemble_error(weights):
            weights = weights / weights.sum()  # Normalize
            ensemble_pred = predictions @ weights
            return np.mean((ensemble_pred - y_val) ** 2)
        
        # Initial guess: equal weights
        w0 = np.ones(len(self.models)) / len(self.models)
        
        # Optimize
        result = minimize(ensemble_error, w0, 
                         constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                         bounds=[(0, 1)] * len(self.models))
        
        self.weights = result.x
        print(f"Optimal ensemble weights: {self.weights}")
    
    def predict(self, image):
        """
        Predict using ensemble
        
        Args:
            image: Input RGB image
        
        Returns:
            Ensemble prediction
        """
        if self.weights is None:
            # Use equal weights if not fitted
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        # Get predictions from each model
        predictions = []
        for model_dict in self.models:
            pred = self._predict_single(model_dict, image)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, weights=self.weights)
        
        return ensemble_pred, predictions
    
    def _predict_single(self, model_dict, image):
        """Helper to predict with a single model"""
        if model_dict['type'] == 'traditional':
            # Traditional TrichromeAnalyzer prediction
            analyzer = model_dict['model']
            _, ml_pred, _ = analyzer.predict_fibrosis(image)
            return ml_pred
        
        elif model_dict['type'] == 'unet':
            # U-Net prediction
            unet_model = model_dict['model']
            mask = predict_with_unet(unet_model, image)
            fibrosis_pct = (np.sum(mask == 1) / mask.size) * 100
            return fibrosis_pct
        
        else:
            raise ValueError(f"Unknown model type: {model_dict['type']}")
    
    def save(self, filepath):
        """Save ensemble configuration"""
        data = {
            'weights': self.weights,
            'model_types': [m['type'] for m in self.models]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)


# ==================== 4. ACTIVE LEARNING PIPELINE ====================

class ActiveLearningPipeline:
    """
    Active learning for efficient model improvement
    
    Strategy: Select most uncertain samples for pathologist review
    """
    
    def __init__(self, model, uncertainty_threshold=0.15):
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertain_samples = []
        self.labeled_pool = []
        self.unlabeled_pool = []
    
    def add_unlabeled_data(self, image_paths):
        """Add unlabeled images to the pool"""
        self.unlabeled_pool.extend(image_paths)
    
    def select_samples_for_labeling(self, n_samples=10, strategy='uncertainty'):
        """
        Select most informative samples for labeling
        
        Args:
            n_samples: Number of samples to select
            strategy: 'uncertainty', 'diversity', or 'hybrid'
        
        Returns:
            List of image paths to label
        """
        if strategy == 'uncertainty':
            return self._uncertainty_sampling(n_samples)
        elif strategy == 'diversity':
            return self._diversity_sampling(n_samples)
        elif strategy == 'hybrid':
            return self._hybrid_sampling(n_samples)
    
    def _uncertainty_sampling(self, n_samples):
        """Select samples with highest prediction uncertainty"""
        uncertainties = []
        
        for img_path in self.unlabeled_pool:
            # Load image
            image = cv2.imread(img_path)
            
            # Get ensemble predictions
            if hasattr(self.model, 'predict'):
                ensemble_pred, individual_preds = self.model.predict(image)
                
                # Uncertainty = std deviation of ensemble predictions
                uncertainty = np.std(individual_preds)
            else:
                # Single model - use features or bootstrap
                uncertainty = self._estimate_uncertainty_single(image)
            
            uncertainties.append((img_path, uncertainty))
        
        # Sort by uncertainty (descending)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        # Select top n
        selected = [path for path, unc in uncertainties[:n_samples]]
        
        # Store uncertain samples
        self.uncertain_samples = uncertainties[:n_samples]
        
        return selected
    
    def _diversity_sampling(self, n_samples):
        """Select diverse samples using feature clustering"""
        from sklearn.cluster import KMeans
        
        # Extract features from all unlabeled samples
        features_list = []
        
        for img_path in self.unlabeled_pool:
            image = cv2.imread(img_path)
            # Extract simple features (color histograms)
            features = self._extract_diversity_features(image)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        # Cluster and select samples closest to cluster centers
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(features_array)
        
        # Find closest sample to each center
        selected_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(features_array - center, axis=1)
            selected_indices.append(np.argmin(distances))
        
        selected = [self.unlabeled_pool[i] for i in selected_indices]
        
        return selected
    
    def _hybrid_sampling(self, n_samples):
        """Combine uncertainty and diversity"""
        n_uncertain = n_samples // 2
        n_diverse = n_samples - n_uncertain
        
        uncertain = self._uncertainty_sampling(n_uncertain)
        
        # Remove uncertain from pool temporarily
        remaining_pool = [p for p in self.unlabeled_pool if p not in uncertain]
        temp_pool = self.unlabeled_pool
        self.unlabeled_pool = remaining_pool
        
        diverse = self._diversity_sampling(n_diverse)
        
        # Restore pool
        self.unlabeled_pool = temp_pool
        
        return uncertain + diverse
    
    def update_with_labels(self, labeled_data):
        """
        Update model with newly labeled data
        
        Args:
            labeled_data: Dict of {image_path: ground_truth_percentage}
        """
        # Move from unlabeled to labeled pool
        for img_path in labeled_data.keys():
            if img_path in self.unlabeled_pool:
                self.unlabeled_pool.remove(img_path)
                self.labeled_pool.append((img_path, labeled_data[img_path]))
        
        # Retrain model with updated data
        print(f"Retraining model with {len(self.labeled_pool)} labeled samples...")
        # (Implementation depends on your model type)
    
    def _estimate_uncertainty_single(self, image):
        """Estimate uncertainty for single model using bootstrap"""
        # Simple heuristic: use feature variance
        # In practice, could use dropout at inference or bootstrap
        return 0.1  # Placeholder
    
    def _extract_diversity_features(self, image):
        """Extract features for diversity sampling"""
        # Color histogram features
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        features = features / features.sum()  # Normalize
        
        return features
    
    def generate_report(self, output_path='active_learning_report.png'):
        """Generate visualization of active learning progress"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Uncertainty distribution
        if self.uncertain_samples:
            uncertainties = [unc for _, unc in self.uncertain_samples]
            axes[0].hist(uncertainties, bins=20, edgecolor='black', alpha=0.7)
            axes[0].axvline(self.uncertainty_threshold, color='red', linestyle='--',
                           label=f'Threshold: {self.uncertainty_threshold}')
            axes[0].set_xlabel('Prediction Uncertainty', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].set_title('Uncertainty Distribution of Unlabeled Samples', fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
        
        # Plot 2: Pool sizes
        pool_sizes = [
            len(self.labeled_pool),
            len(self.unlabeled_pool),
            len(self.uncertain_samples)
        ]
        labels = ['Labeled', 'Unlabeled', 'High Uncertainty']
        colors = ['#48bb78', '#4299e1', '#f56565']
        
        axes[1].bar(labels, pool_sizes, color=colors, edgecolor='black')
        axes[1].set_ylabel('Number of Samples', fontsize=12)
        axes[1].set_title('Active Learning Data Pools', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (label, size) in enumerate(zip(labels, pool_sizes)):
            axes[1].text(i, size + max(pool_sizes)*0.02, str(size),
                        ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Active learning report saved to: {output_path}")


# ==================== INTEGRATION EXAMPLE ====================

def create_advanced_pipeline():
    """
    Example: Create complete pipeline with all advanced features
    """
    print("Creating Advanced Trichrome Analysis Pipeline")
    print("="*70)
    
    # 1. Color Normalization
    print("\n1. Setting up color normalization...")
    normalizer = MacenkoNormalizer()
    # Fit to reference image
    # reference_image = cv2.imread('reference_trichrome.jpg')
    # normalizer.fit(reference_image)
    # normalizer.save('macenko_normalizer.pkl')
    
    # 2. U-Net Model
    print("2. Initializing U-Net model...")
    unet_model = UNet(in_channels=3, out_channels=3, init_features=32)
    # Load pretrained if available
    # unet_model.load_state_dict(torch.load('unet_best.pth'))
    
    # 3. Ensemble
    print("3. Creating ensemble...")
    ensemble = EnsemblePredictor()
    # ensemble.add_model(traditional_analyzer, 'traditional')
    # ensemble.add_model(unet_model, 'unet')
    
    # 4. Active Learning
    print("4. Setting up active learning...")
    active_learner = ActiveLearningPipeline(ensemble)
    
    print("\n✓ Pipeline ready!")
    
    return {
        'normalizer': normalizer,
        'unet': unet_model,
        'ensemble': ensemble,
        'active_learner': active_learner
    }


if __name__ == "__main__":
    # Example usage
    pipeline = create_advanced_pipeline()
    
    print("\nAdvanced features ready to use!")
    print("\nNext steps:")
    print("1. Fit color normalizer to reference image")
    print("2. Prepare training data and train U-Net")
    print("3. Add models to ensemble and fit weights")
    print("4. Start active learning loop")
