"""
Trichrome Fibrosis Analyzer - Core Engine with Vahadane Normalization
====================================================================

Complete analysis pipeline with Vahadane color normalization, segmentation, 
ML models, and explainability.

Usage:
    from trichrome_core import TrichromeAnalyzer
    
    analyzer = TrichromeAnalyzer()
    analyzer.train(training_folder, use_normalization=True)
    analyzer.save_model('model.pkl')
    
    # Predict
    analyzer.load_model('model.pkl')
    raw, ml_pred, mask = analyzer.predict(image)
"""

import os
import re
import cv2
import numpy as np
import pickle
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, cohen_kappa_score
from scipy.stats import pearsonr
from scipy.optimize import nnls


# ==================== VAHADANE COLOR NORMALIZATION ====================

class VahadaneNormalizer:
    """
    Stain normalization using Vahadane method with sparse NMF.
    Better for trichrome as it handles multiple stain components.
    """
    
    def __init__(self, lambda_reg=0.1, max_iter=20):
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.target_stains = None
        self.target_concentrations = None
        self.maxC_target = None
    
    def fit(self, target_image):
        """Fit to reference image (RGB, 0-255)"""
        od = self._rgb_to_od(target_image)
        
        # Subsample for speed - use only 10,000 pixels
        od_flat = od.reshape(-1, 3)
        transparent_mask = self._is_transparent(od_flat)
        od_hat = od_flat[~transparent_mask]
        
        # Subsample if too many pixels
        if od_hat.shape[0] > 10000:
            indices = np.random.choice(od_hat.shape[0], 10000, replace=False)
            od_hat = od_hat[indices]
        
        # Extract stain matrix using dictionary learning
        self.target_stains = self._get_stain_matrix(od_hat)
        self.target_concentrations = self._get_concentrations(od, self.target_stains)
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=(0, 1))
        
        return self
    
    def transform(self, image):
        """Normalize image to match reference"""
        if self.target_stains is None:
            raise ValueError("Not fitted. Call fit() first.")
        
        od = self._rgb_to_od(image)
        
        # Subsample for stain extraction - use only 5,000 pixels
        od_flat = od.reshape(-1, 3)
        transparent_mask = self._is_transparent(od_flat)
        od_hat = od_flat[~transparent_mask]
        
        # Subsample if too many pixels
        if od_hat.shape[0] > 5000:
            indices = np.random.choice(od_hat.shape[0], 5000, replace=False)
            od_hat = od_hat[indices]
        
        # Extract stain matrix for source image
        stains = self._get_stain_matrix(od_hat)
        concentrations = self._get_concentrations(od, stains)
        
        # Normalize concentration statistics
        maxC = np.percentile(concentrations, 99, axis=(0, 1))
        concentrations *= (self.maxC_target / (maxC + 1e-6))
        
        # Reconstruct with target stains: od = concentrations @ target_stains.T
        # concentrations is (H, W, n_stains), target_stains is (3, n_stains)
        h, w = concentrations.shape[:2]
        concentrations_flat = concentrations.reshape(-1, concentrations.shape[2])
        normalized_od_flat = concentrations_flat @ self.target_stains.T
        normalized_od = normalized_od_flat.reshape(h, w, 3)
        normalized = self._od_to_rgb(normalized_od)
        
        return normalized.astype(np.uint8)
    
    def _rgb_to_od(self, image):
        """Convert RGB to optical density"""
        image = image.astype(np.float64)
        image = np.maximum(image, 1)
        return -np.log(image / 255.0)
    
    def _od_to_rgb(self, od):
        """Convert optical density to RGB"""
        rgb = np.exp(-od) * 255
        return np.clip(rgb, 0, 255)
    
    def _is_transparent(self, od_flat, threshold=0.15):
        """Check if pixel is background/transparent (expects flattened OD)"""
        return np.all(od_flat < threshold, axis=1)
    
    def _get_stain_matrix(self, od_hat, n_stains=2):
        """
        Extract stain matrix using sparse dictionary learning.
        Uses non-negative least squares for sparse decomposition.
        
        Args:
            od_hat: (n_pixels, 3) - optical density of non-transparent pixels
            
        Returns:
            stains: (3, n_stains) - each column is a stain vector in RGB space
        """
        # Initialize with SVD
        # We want to decompose od_hat (n_pixels, 3) ≈ concentrations (n_pixels, n_stains) @ stains.T (n_stains, 3)
        # So stains should be (3, n_stains)
        
        # SVD on transposed od_hat
        U, S, Vt = np.linalg.svd(od_hat.T, full_matrices=False)  # od_hat.T is (3, n_pixels)
        # U is (3, 3), we take first n_stains columns -> (3, n_stains)
        stains = np.abs(U[:, :n_stains])  # (3, n_stains)
        
        # Refine with alternating NNLS (reduced iterations)
        for iteration in range(5):  # Only 5 iterations for speed
            # Update concentrations: od_hat[i] ≈ stains @ concentrations[i]
            concentrations = np.zeros((od_hat.shape[0], n_stains))
            for i in range(od_hat.shape[0]):
                concentrations[i], _ = nnls(stains, od_hat[i])
            
            # Update stains: od_hat[:, rgb] ≈ concentrations @ stains[rgb, :]
            stains_new = np.zeros((3, n_stains))
            for rgb_channel in range(3):
                stains_new[rgb_channel, :], _ = nnls(
                    concentrations, 
                    od_hat[:, rgb_channel]
                )
            
            stains = stains_new
            
            # Normalize each stain vector
            for j in range(n_stains):
                norm = np.linalg.norm(stains[:, j])
                if norm > 1e-6:
                    stains[:, j] /= norm
        
        return stains
    
    def _get_concentrations(self, od, stains):
        """Get concentration matrix using pseudo-inverse (much faster than NNLS per pixel)
        
        Args:
            od: Optical density (H, W, 3)
            stains: Stain matrix (3, n_stains)
        
        Returns:
            concentrations: (H, W, n_stains)
        """
        h, w = od.shape[:2]
        od_flat = od.reshape(-1, 3)
        
        # Use least squares instead of NNLS for speed (then clip negatives)
        # Solve: od = stains @ concentrations.T
        # concentrations = (stains.T @ stains)^-1 @ stains.T @ od.T
        concentrations_flat = np.linalg.lstsq(stains, od_flat.T, rcond=None)[0].T
        
        # Clip negative values to 0 (since concentrations should be non-negative)
        concentrations_flat = np.maximum(concentrations_flat, 0)
        
        return concentrations_flat.reshape(h, w, -1)
    
    def save(self, filepath):
        """Save normalizer parameters"""
        data = {
            'target_stains': self.target_stains,
            'target_concentrations': self.target_concentrations,
            'maxC_target': self.maxC_target,
            'lambda_reg': self.lambda_reg,
            'max_iter': self.max_iter
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """Load normalizer parameters"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.target_stains = data['target_stains']
        self.target_concentrations = data['target_concentrations']
        self.maxC_target = data['maxC_target']
        self.lambda_reg = data.get('lambda_reg', 0.1)
        self.max_iter = data.get('max_iter', 100)
        return self


# ==================== TRICHROME ANALYZER ====================

class TrichromeAnalyzer:
    """Main analyzer with segmentation and ML capabilities"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.normalizer = None
        
    # ==================== SEGMENTATION ====================
    
    def detect_red_pink_glomeruli(self, image):
        """Detect RED/PINK glomeruli (healthy)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        lower_red1 = np.array([0, 20, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 20, 50])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_pink_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        blurred = cv2.GaussianBlur(red_pink_mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=50, param2=30, minRadius=15, maxRadius=120
        )
        
        red_glomeruli_mask = np.zeros(gray.shape, dtype=np.uint8)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = int(circle[2] * 1.1)
                cv2.circle(red_glomeruli_mask, center, radius, 255, -1)
        
        return red_glomeruli_mask > 0
    
    def detect_blue_green_glomeruli(self, image):
        """Detect BLUE/GREEN glomeruli (sclerosed)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        lower_blue_green = np.array([80, 30, 50])
        upper_blue_green = np.array([170, 255, 255])
        blue_green_mask = cv2.inRange(hsv, lower_blue_green, upper_blue_green)
        
        blurred = cv2.GaussianBlur(blue_green_mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=50, param2=30, minRadius=15, maxRadius=120
        )
        
        blue_glomeruli_mask = np.zeros(gray.shape, dtype=np.uint8)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = int(circle[2] * 1.1)
                cv2.circle(blue_glomeruli_mask, center, radius, 255, -1)
        
        return blue_glomeruli_mask > 0
    
    def segment_fibrosis(self, image):
        """Segment interstitial fibrosis - detects BLUE collagen fibers"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create tissue mask (exclude white background)
        _, tissue_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        
        # Detect blue/cyan collagen (TRICHROME FIBROSIS - this is what we want!)
        # Hue: 90-130 is blue/cyan range
        # Sat: >40 to avoid pale/white areas
        # Val: >30 to avoid very dark areas
        lower_blue = np.array([90, 40, 30])
        upper_blue = np.array([130, 255, 255])
        fibrosis_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Only keep fibrosis within tissue (exclude background)
        fibrosis_mask = cv2.bitwise_and(fibrosis_mask, tissue_mask)
        
        # Exclude glomeruli (both red and blue types)
        red_glomeruli = self.detect_red_pink_glomeruli(image)
        blue_glomeruli = self.detect_blue_green_glomeruli(image)
        
        fibrosis_mask[red_glomeruli] = 0
        fibrosis_mask[blue_glomeruli] = 0
        
        # Morphological cleanup
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        
        # Remove small noise
        fibrosis_mask = cv2.morphologyEx(fibrosis_mask, cv2.MORPH_OPEN, 
                                         kernel_small, iterations=1)
        # Fill small holes
        fibrosis_mask = cv2.morphologyEx(fibrosis_mask, cv2.MORPH_CLOSE, 
                                         kernel_medium, iterations=1)
        
        return fibrosis_mask
    
    def extract_features(self, image, mask):
        """Extract 14 features for ML"""
        features = []
        
        total_pixels = mask.size
        fibrosis_pixels = np.sum(mask > 0)
        fibrosis_ratio = fibrosis_pixels / total_pixels
        features.append(fibrosis_ratio * 100)
        
        if fibrosis_pixels > 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            h_vals = hsv[:, :, 0][mask > 0]
            s_vals = hsv[:, :, 1][mask > 0]
            v_vals = hsv[:, :, 2][mask > 0]
            
            features.extend([
                np.mean(h_vals), np.std(h_vals),
                np.mean(s_vals), np.std(s_vals),
                np.mean(v_vals),
            ])
            
            b_vals = lab[:, :, 2][mask > 0]
            features.extend([np.mean(b_vals), np.std(b_vals)])
            
            y_coords, x_coords = np.where(mask > 0)
            features.extend([
                np.std(x_coords) / image.shape[1],
                np.std(y_coords) / image.shape[0],
            ])
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            num_regions = num_labels - 1
            
            if num_regions > 0:
                areas = stats[1:, cv2.CC_STAT_AREA]
                features.extend([
                    num_regions,
                    np.mean(areas),
                    np.std(areas) if len(areas) > 1 else 0,
                    np.max(areas),
                ])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0] * 13)
        
        return np.array(features)
    
    # ==================== TRAINING ====================
    
    def train(self, training_folder, model_type='auto', use_normalization=True, 
              reference_image=None):
        """
        Train model on labeled images from folder
        
        Args:
            training_folder: Path to folder with images named like "tri_15.5%.jpg"
            model_type: 'Ridge', 'RandomForest', 'GradientBoosting', or 'auto'
            use_normalization: Whether to apply Vahadane color normalization
            reference_image: Path to reference image for normalization (if None, uses first image)
        
        Returns:
            Training metrics dict
        """
        # Setup color normalization if requested
        if use_normalization:
            print("\n🎨 Setting up Vahadane color normalization...")
            if self.normalizer is None:
                self.normalizer = VahadaneNormalizer()
                
                if reference_image is None:
                    # Use first training image as reference
                    first_image_path = self._find_trichrome_images(training_folder)[0]
                    ref_img = cv2.imread(first_image_path)
                    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                else:
                    ref_img = cv2.imread(reference_image)
                    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                
                print("  Fitting normalizer to reference image...")
                self.normalizer.fit(ref_img)
                print("✅ Color normalization fitted")
        
        X, y, paths = self._prepare_training_data(training_folder)
        
        if len(X) < 5:
            print(f"Warning: Only {len(X)} samples - need more for reliable model")
        
        X_scaled = self.scaler.fit_transform(X)
        
        models = {
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
            )
        }
        
        if model_type != 'auto' and model_type in models:
            self.model = models[model_type]
            self.model.fit(X_scaled, y)
            selected_name = model_type
        else:
            # Cross-validation to select best
            print("\n=== Cross-Validation ===")
            best_score = -float('inf')
            selected_name = None
            
            for name, model in models.items():
                scores = cross_val_score(
                    model, X_scaled, y, cv=min(5, len(X)),
                    scoring='neg_mean_absolute_error', n_jobs=-1
                )
                mean_mae = -scores.mean()
                print(f"{name}: MAE = {mean_mae:.2f}% ± {scores.std():.2f}%")
                
                if -scores.mean() > best_score:
                    best_score = -scores.mean()
                    selected_name = name
            
            self.model = models[selected_name]
            self.model.fit(X_scaled, y)
        
        # Training metrics
        train_pred = self.model.predict(X_scaled)
        metrics = {
            'model_type': selected_name,
            'n_samples': len(X),
            'mae': mean_absolute_error(y, train_pred),
            'r2': r2_score(y, train_pred)
        }
        
        print(f"\n=== Trained: {selected_name} ===")
        print(f"Training MAE: {metrics['mae']:.2f}%")
        print(f"Training R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def _prepare_training_data(self, training_folder):
        """Find and process training images"""
        image_files = self._find_trichrome_images(training_folder)
        
        X, y, valid_paths = [], [], []
        
        for img_path in image_files:
            percentage = self._extract_percentage(os.path.basename(img_path))
            if percentage is None:
                continue
            
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Apply color normalization if enabled
            if self.normalizer is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb = self.normalizer.transform(image_rgb)
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Resize if needed
            image = self._resize_if_needed(image)
            
            fibrosis_mask = self.segment_fibrosis(image)
            features = self.extract_features(image, fibrosis_mask)
            
            X.append(features)
            y.append(percentage)
            valid_paths.append(img_path)
            
            raw_pct = (np.sum(fibrosis_mask > 0) / fibrosis_mask.size) * 100
            print(f"Processed: {os.path.basename(img_path)} - "
                  f"Label: {percentage}% | Raw: {raw_pct:.1f}%")
        
        return np.array(X), np.array(y), valid_paths
    
    def _find_trichrome_images(self, folder):
        """Find all trichrome images"""
        trichrome_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().startswith('tri') and file.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
                ):
                    full_path = os.path.join(root, file)
                    trichrome_files.append(full_path)
        
        print(f"Found {len(trichrome_files)} trichrome images")
        return trichrome_files
    
    def _extract_percentage(self, filename):
        """Extract percentage from filename"""
        match = re.search(r'(\d+\.?\d*)\s*%?', filename)
        if match:
            return float(match.group(1))
        return None
    
    def _resize_if_needed(self, image, max_dim=1024):
        """Resize image if too large"""
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        return image
    
    # ==================== PREDICTION ====================
    
    def predict(self, image, return_timings=False):
        """
        Predict fibrosis percentage
        
        Args:
            image: BGR image (numpy array) or path to image
            return_timings: Whether to return performance timings
        
        Returns:
            raw_percent, ml_prediction, fibrosis_mask, [timings]
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if return_timings:
            timings = {}
            total_start = time.time()
            t0 = time.time()
        
        # Apply color normalization if fitted
        if self.normalizer is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = self.normalizer.transform(image_rgb)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if return_timings:
                timings['normalization'] = (time.time() - t0) * 1000
                t0 = time.time()
        
        # Resize
        image = self._resize_if_needed(image)
        
        if return_timings:
            timings['resize'] = (time.time() - t0) * 1000
            t0 = time.time()
        
        # Segment
        fibrosis_mask = self.segment_fibrosis(image)
        raw_percent = (np.sum(fibrosis_mask > 0) / fibrosis_mask.size) * 100
        
        if return_timings:
            timings['segmentation'] = (time.time() - t0) * 1000
            t0 = time.time()
        
        # ML prediction
        if self.model is not None:
            features = self.extract_features(image, fibrosis_mask)
            features_scaled = self.scaler.transform([features])
            ml_prediction = self.model.predict(features_scaled)[0]
            ml_prediction = np.clip(ml_prediction, 0, 100)
        else:
            ml_prediction = None
        
        if return_timings:
            timings['ml_inference'] = (time.time() - t0) * 1000
            timings['total'] = (time.time() - total_start) * 1000
            return raw_percent, ml_prediction, fibrosis_mask, timings
        
        return raw_percent, ml_prediction, fibrosis_mask
    
    # ==================== EXPLAINABILITY ====================
    
    def generate_heatmap(self, image, fibrosis_mask):
        """Generate severity heatmap overlay"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        heatmap = np.zeros_like(image, dtype=np.float32)
        mask_bool = fibrosis_mask > 0
        
        if np.any(mask_bool):
            sat_normalized = np.zeros_like(saturation, dtype=np.uint8)
            sat_values = saturation[mask_bool]
            if sat_values.max() > sat_values.min():
                sat_normalized[mask_bool] = (
                    (saturation[mask_bool] - sat_values.min()) /
                    (sat_values.max() - sat_values.min()) * 255
                ).astype(np.uint8)
            
            heatmap_colored = cv2.applyColorMap(sat_normalized, cv2.COLORMAP_JET)
            heatmap[mask_bool] = heatmap_colored[mask_bool]
        
        overlay = cv2.addWeighted(image.astype(np.float32), 0.6, heatmap, 0.4, 0)
        return overlay.astype(np.uint8)
    
    def explain_features(self, image):
        """
        Explain prediction with feature contributions
        
        Returns:
            dict with prediction, features, and top contributors
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        fibrosis_mask = self.segment_fibrosis(image)
        features = self.extract_features(image, fibrosis_mask)
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        
        feature_names = [
            'Area %', 'Hue Mean', 'Hue Std', 'Sat Mean', 'Sat Std',
            'Value Mean', 'LAB_B Mean', 'LAB_B Std', 'Spread X',
            'Spread Y', 'Num Regions', 'Avg Region Size',
            'Region Size Std', 'Max Region Size'
        ]
        
        explanation = {'prediction': prediction, 'features': {}}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for name, feat_val, importance in zip(feature_names, features, importances):
                explanation['features'][name] = {
                    'value': float(feat_val),
                    'importance': float(importance),
                    'contribution': float(feat_val * importance)
                }
        elif hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_
            for name, feat_val, coef in zip(feature_names, features, coefficients):
                explanation['features'][name] = {
                    'value': float(feat_val),
                    'coefficient': float(coef),
                    'contribution': float(feat_val * coef)
                }
        
        sorted_features = sorted(
            explanation['features'].items(),
            key=lambda x: abs(x[1].get('contribution', 0)),
            reverse=True
        )
        
        explanation['top_features'] = [
            {'name': name, **data} for name, data in sorted_features[:5]
        ]
        
        return explanation
    
    # ==================== MODEL I/O ====================
    
    def save_model(self, filepath='trichrome_model.pkl'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'normalizer': self.normalizer
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='trichrome_model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.normalizer = data.get('normalizer', None)
        print(f"✓ Model loaded from {filepath}")


# ==================== CLINICAL VALIDATION ====================

def validate_clinical(analyzer, test_csv_path):
    """
    Perform clinical validation study
    
    Args:
        analyzer: Trained TrichromeAnalyzer
        test_csv_path: CSV with columns 'filepath', 'ground_truth'
    
    Returns:
        dict with validation metrics
    """
    import pandas as pd
    
    df = pd.read_csv(test_csv_path)
    
    predictions = []
    ground_truth = df['ground_truth'].values
    
    print(f"\nValidating on {len(df)} images...")
    
    for i, row in df.iterrows():
        image = cv2.imread(row['filepath'])
        _, pred, _ = analyzer.predict(image)
        predictions.append(pred if pred is not None else 0)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(df)}")
    
    predictions = np.array(predictions)
    
    mae = mean_absolute_error(ground_truth, predictions)
    r2 = r2_score(ground_truth, predictions)
    pearson_r, pearson_p = pearsonr(ground_truth, predictions)
    
    differences = predictions - ground_truth
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    def categorize(percent):
        if percent < 10: return "Minimal"
        elif percent < 25: return "Mild"
        elif percent < 50: return "Moderate"
        else: return "Severe"
    
    categories_true = [categorize(x) for x in ground_truth]
    categories_pred = [categorize(x) for x in predictions]
    kappa = cohen_kappa_score(categories_true, categories_pred)
    
    print("\n" + "="*50)
    print("CLINICAL VALIDATION RESULTS")
    print("="*50)
    print(f"MAE: {mae:.2f}%")
    print(f"R²: {r2:.3f}")
    print(f"Pearson r: {pearson_r:.3f} (p={pearson_p:.4f})")
    print(f"Bias: {mean_diff:.2f}% ± {std_diff:.2f}%")
    print(f"Cohen's Kappa: {kappa:.3f}")
    
    return {
        'mae': mae, 'r2': r2, 'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'bias': mean_diff, 'std_diff': std_diff, 'kappa': kappa,
        'predictions': predictions, 'ground_truth': ground_truth
    }


# ==================== BATCH PROCESSING ====================

def batch_process(analyzer, input_folder, output_csv='results.csv'):
    """
    Process all images in folder and save results
    
    Args:
        analyzer: Trained TrichromeAnalyzer
        input_folder: Folder with images to process
        output_csv: Where to save results
    """
    import pandas as pd
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).rglob(ext))
    
    print(f"\nBatch processing {len(image_files)} images...")
    
    results = []
    
    for i, img_path in enumerate(image_files):
        try:
            image = cv2.imread(str(img_path))
            raw, ml_pred, _ = analyzer.predict(image)
            
            results.append({
                'filename': img_path.name,
                'filepath': str(img_path),
                'raw_percent': raw,
                'ml_prediction': ml_pred if ml_pred is not None else raw,
                'status': 'success'
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(image_files)}")
        
        except Exception as e:
            results.append({
                'filename': img_path.name,
                'filepath': str(img_path),
                'raw_percent': None,
                'ml_prediction': None,
                'status': f'error: {str(e)}'
            })
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Results saved to {output_csv}")
    print(f"  Successful: {len(df[df.status == 'success'])}/{len(df)}")
    
    return df


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trichrome Fibrosis Analyzer')
    parser.add_argument('command', choices=['train', 'predict', 'validate', 'batch'],
                       help='Command to execute')
    parser.add_argument('--data', type=str, help='Data directory or image path')
    parser.add_argument('--model', type=str, default='trichrome_model.pkl',
                       help='Model file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable color normalization')
    
    args = parser.parse_args()
    
    analyzer = TrichromeAnalyzer()
    
    if args.command == 'train':
        print("Training model...")
        analyzer.train(args.data, use_normalization=not args.no_normalize)
        analyzer.save_model(args.model)
    
    elif args.command == 'predict':
        analyzer.load_model(args.model)
        raw, ml, mask = analyzer.predict(args.data)
        print(f"\nResults for {args.data}:")
        print(f"  Raw: {raw:.2f}%")
        print(f"  ML:  {ml:.2f}%")
    
    elif args.command == 'validate':
        analyzer.load_model(args.model)
        validate_clinical(analyzer, args.data)
    
    elif args.command == 'batch':
        analyzer.load_model(args.model)
        output = args.output or 'batch_results.csv'
        batch_process(analyzer, args.data, output)