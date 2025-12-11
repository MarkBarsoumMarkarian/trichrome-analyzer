import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, cohen_kappa_score
from scipy.stats import pearsonr
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import pandas as pd

class TrichromeAnalyzer:
    def __init__(self, base_folder):
        self.base_folder = Path(base_folder)
        self.model = None
        self.scaler = StandardScaler()
        self.images_data = []
        
    def find_trichrome_images(self):
        """Recursively find all trichrome images in folder structure"""
        trichrome_files = []
        
        for root, dirs, files in os.walk(self.base_folder):
            for file in files:
                # Check if file starts with 'tri' and is an image
                if file.lower().startswith('tri') and file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    full_path = os.path.join(root, file)
                    trichrome_files.append(full_path)
        
        print(f"Found {len(trichrome_files)} trichrome images")
        return trichrome_files
    
    def extract_percentage(self, filename):
        """Extract percentage from filename (e.g., 'tri 15%' -> 15.0)"""
        # Look for patterns like "15%", "15 %", or just "15"
        match = re.search(r'(\d+\.?\d*)\s*%?', filename)
        if match:
            return float(match.group(1))
        return None
    
    def detect_glomeruli(self, image):
        """
        Legacy wrapper - detects ALL glomeruli (both red and blue)
        For compatibility with old code
        """
        red_glom = self.detect_red_pink_glomeruli(image)
        blue_glom = self.detect_blue_green_glomeruli(image)
        return np.logical_or(red_glom, blue_glom)
    
    def detect_red_pink_glomeruli(self, image):
        """
        Detect RED/PINK glomeruli (healthy glomeruli with normal structure)
        These are the regions AROUND which we want to measure fibrosis
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect red/pink tissue
        lower_red1 = np.array([0, 20, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 20, 50])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_pink_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Find circular structures in red/pink areas
        blurred = cv2.GaussianBlur(red_pink_mask, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=120
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
        """
        Detect BLUE/GREEN glomeruli (sclerosed/abnormal glomeruli)
        These should be IGNORED from fibrosis calculation
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect blue/green tissue
        lower_blue_green = np.array([80, 30, 50])
        upper_blue_green = np.array([170, 255, 255])
        blue_green_mask = cv2.inRange(hsv, lower_blue_green, upper_blue_green)
        
        # Find circular structures in blue/green areas
        blurred = cv2.GaussianBlur(blue_green_mask, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=120
        )
        
        blue_glomeruli_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = int(circle[2] * 1.1)
                cv2.circle(blue_glomeruli_mask, center, radius, 255, -1)
        
        return blue_glomeruli_mask > 0
    
    def segment_fibrosis(self, image, exclude_glomeruli=True):
        """
        Segment interstitial fibrosis from trichrome stain
        
        COMBINED APPROACH (Options A + B):
        1. Detect ALL blue/green tissue (Option A)
        2. Create large ROI around red glomeruli (Option B - expanded)
        3. Exclude blue/green glomeruli
        4. Use UNION of both approaches for robustness
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find ALL tissue (exclude white background)
        _, tissue_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # ===== OPTION A: Detect ALL blue/green tissue globally =====
        # Very permissive - capture all blue/green/purple regardless of location
        lower_blue_green = np.array([75, 15, 30])   # Even wider range
        upper_blue_green = np.array([170, 255, 255])
        
        all_blue_green = cv2.inRange(hsv, lower_blue_green, upper_blue_green)
        all_blue_green = cv2.bitwise_and(all_blue_green, tissue_mask)
        
        # ===== OPTION B: Focus on areas around red glomeruli (expanded ROI) =====
        red_glomeruli = self.detect_red_pink_glomeruli(image)
        
        # Much larger expansion (100x100 instead of 50x50)
        kernel_expand = np.ones((100, 100), np.uint8)
        red_glomeruli_expanded = cv2.dilate(red_glomeruli.astype(np.uint8) * 255, 
                                            kernel_expand, iterations=1)
        
        # Blue/green tissue within expanded ROI
        roi_blue_green = cv2.bitwise_and(all_blue_green, all_blue_green,
                                         mask=red_glomeruli_expanded)
        
        # ===== COMBINE: Union of both approaches =====
        # This ensures we don't miss fibrosis even if glomeruli detection fails
        fibrosis_mask = cv2.bitwise_or(all_blue_green, roi_blue_green)
        
        # ===== EXCLUSIONS =====
        # 1. Remove blue/green glomeruli (sclerosed - abnormal)
        blue_glomeruli = self.detect_blue_green_glomeruli(image)
        fibrosis_mask[blue_glomeruli] = 0
        
        # 2. Remove red/pink glomeruli themselves (keep surrounding area)
        fibrosis_mask[red_glomeruli] = 0
        
        # 3. Exclude very light/white areas
        light_mask = gray > 190
        fibrosis_mask[light_mask] = 0
        
        # ===== MORPHOLOGICAL CLEANUP =====
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((4, 4), np.uint8)
        
        fibrosis_mask = cv2.morphologyEx(fibrosis_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        fibrosis_mask = cv2.morphologyEx(fibrosis_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        return fibrosis_mask
    
    def extract_features(self, image, mask):
        """Extract comprehensive features from segmented fibrosis"""
        features = []
        
        # Basic area ratio
        total_pixels = mask.size
        fibrosis_pixels = np.sum(mask > 0)
        fibrosis_ratio = fibrosis_pixels / total_pixels
        features.append(fibrosis_ratio * 100)  # As percentage
        
        if fibrosis_pixels > 0:
            # Color features in multiple color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # HSV stats
            h_vals = hsv[:, :, 0][mask > 0]
            s_vals = hsv[:, :, 1][mask > 0]
            v_vals = hsv[:, :, 2][mask > 0]
            
            features.extend([
                np.mean(h_vals),
                np.std(h_vals),
                np.mean(s_vals),
                np.std(s_vals),
                np.mean(v_vals),
            ])
            
            # LAB stats (particularly B channel for blue)
            b_vals = lab[:, :, 2][mask > 0]
            features.extend([
                np.mean(b_vals),
                np.std(b_vals),
            ])
            
            # Spatial features
            y_coords, x_coords = np.where(mask > 0)
            features.extend([
                np.std(x_coords) / image.shape[1],  # Normalized spread
                np.std(y_coords) / image.shape[0],
            ])
            
            # Texture features
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
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
            # No fibrosis detected
            features.extend([0] * 13)
        
        return np.array(features)
    
    def prepare_training_data(self):
        """Prepare training data from labeled images"""
        image_files = self.find_trichrome_images()
        
        X = []
        y = []
        valid_images = []
        
        for img_path in image_files:
            percentage = self.extract_percentage(os.path.basename(img_path))
            
            if percentage is None:
                print(f"Warning: Could not extract percentage from {img_path}")
                continue
            
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            # Resize large images for consistency
            max_dim = 1024
            h, w = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image = cv2.resize(image, None, fx=scale, fy=scale)
            
            fibrosis_mask = self.segment_fibrosis(image)
            features = self.extract_features(image, fibrosis_mask)
            
            X.append(features)
            y.append(percentage)
            valid_images.append(img_path)
            
            # Calculate raw percentage for comparison
            raw_pct = (np.sum(fibrosis_mask > 0) / fibrosis_mask.size) * 100
            print(f"Processed: {os.path.basename(img_path)} - Label: {percentage}% | Raw segmentation: {raw_pct:.1f}%")
        
        self.images_data = list(zip(valid_images, X, y))
        return np.array(X), np.array(y), valid_images
    
    def train_model(self, X, y):
        """Train regression model with cross-validation"""
        if len(X) < 5:
            print(f"Warning: Only {len(X)} training samples. Need more data for reliable model.")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Try multiple models
        models = {
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=5,  # Reduced to prevent overfitting
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Evaluate models with cross-validation
        print("\n=== Cross-Validation Results ===")
        best_score = -float('inf')
        best_model_name = None
        
        for name, model in models.items():
            scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X)), 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
            mean_mae = -scores.mean()
            std_mae = scores.std()
            print(f"{name}: MAE = {mean_mae:.2f}% ± {std_mae:.2f}%")
            
            if -scores.mean() > best_score:
                best_score = -scores.mean()
                best_model_name = name
        
        # Train best model on all data
        self.model = models[best_model_name]
        self.model.fit(X_scaled, y)
        
        print(f"\n=== Selected Model: {best_model_name} ===")
        
        # Final evaluation
        train_pred = self.model.predict(X_scaled)
        
        print(f"Training MAE: {mean_absolute_error(y, train_pred):.2f}%")
        print(f"Training R²: {r2_score(y, train_pred):.3f}")
        
        # Show feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            print("\nTop 3 Most Important Features:")
            importances = self.model.feature_importances_
            top_idx = np.argsort(importances)[-3:][::-1]
            feature_names = ['Area%', 'H_mean', 'H_std', 'S_mean', 'S_std', 'V_mean',
                           'LAB_B_mean', 'LAB_B_std', 'Spread_X', 'Spread_Y',
                           'Num_regions', 'Avg_region', 'Std_region', 'Max_region']
            for idx in top_idx:
                print(f"  {feature_names[idx]}: {importances[idx]:.3f}")
        
        return self.model
    
    def predict_fibrosis(self, image_path):
        """Predict fibrosis percentage for a new image"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize if needed
        max_dim = 1024
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        fibrosis_mask = self.segment_fibrosis(image, exclude_glomeruli=True)
        features = self.extract_features(image, fibrosis_mask)
        
        # Scale features and predict
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        
        # Ensure prediction is within reasonable bounds
        prediction = np.clip(prediction, 0, 100)
        
        return prediction, fibrosis_mask
    
    def visualize_analysis(self, image_path, save_path=None):
        """Visualize the analysis process"""
        image = cv2.imread(image_path)
        
        # Resize if needed
        max_dim = 1024
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        glomeruli_mask = self.detect_glomeruli(image)
        fibrosis_mask = self.segment_fibrosis(image, exclude_glomeruli=True)
        
        # Calculate raw percentage
        raw_pct = (np.sum(fibrosis_mask > 0) / fibrosis_mask.size) * 100
        
        # Get prediction if model is trained
        if self.model is not None:
            predicted_percentage, _ = self.predict_fibrosis(image_path)
            title_suffix = f"\nRaw: {raw_pct:.1f}% | ML Predicted: {predicted_percentage:.1f}%"
        else:
            title_suffix = f"\nRaw: {raw_pct:.1f}%"
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Trichrome Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Glomeruli detection
        glomeruli_overlay = image_rgb.copy()
        glomeruli_overlay[glomeruli_mask] = [255, 0, 0]
        blended_glom = cv2.addWeighted(image_rgb, 0.6, glomeruli_overlay, 0.4, 0)
        axes[0, 1].imshow(blended_glom)
        axes[0, 1].set_title('Glomeruli Detection (Excluded from Analysis)', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Fibrosis segmentation
        axes[1, 0].imshow(fibrosis_mask, cmap='gray')
        axes[1, 0].set_title('Segmented Interstitial Fibrosis', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Overlay
        overlay = np.zeros_like(image_rgb)
        overlay[fibrosis_mask > 0] = [0, 255, 255]
        blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
        axes[1, 1].imshow(blended)
        axes[1, 1].set_title(f'Fibrosis Overlay{title_suffix}', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def analyze_color_distribution(self, image_path, sample_points=1000):
        """
        Analyze color distribution to help calibrate thresholds
        Shows HSV values for fibrotic vs normal tissue
        """
        image = cv2.imread(image_path)
        if image is None:
            return
        
        # Resize if needed
        max_dim = 1024
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sample random tissue pixels (exclude background)
        tissue_mask = gray < 200
        tissue_coords = np.column_stack(np.where(tissue_mask))
        
        if len(tissue_coords) > sample_points:
            indices = np.random.choice(len(tissue_coords), sample_points, replace=False)
            sampled = tissue_coords[indices]
        else:
            sampled = tissue_coords
        
        h_vals = [hsv[y, x, 0] for y, x in sampled]
        s_vals = [hsv[y, x, 1] for y, x in sampled]
        v_vals = [hsv[y, x, 2] for y, x in sampled]
        gray_vals = [gray[y, x] for y, x in sampled]
        
        print(f"\n=== Color Analysis for {os.path.basename(image_path)} ===")
        print(f"Hue (H):        Mean={np.mean(h_vals):.1f}, Std={np.std(h_vals):.1f}, Range=[{np.min(h_vals)}-{np.max(h_vals)}]")
        print(f"Saturation (S): Mean={np.mean(s_vals):.1f}, Std={np.std(s_vals):.1f}, Range=[{np.min(s_vals)}-{np.max(s_vals)}]")
        print(f"Value (V):      Mean={np.mean(v_vals):.1f}, Std={np.std(v_vals):.1f}, Range=[{np.min(v_vals)}-{np.max(v_vals)}]")
        print(f"Grayscale:      Mean={np.mean(gray_vals):.1f}, Std={np.std(gray_vals):.1f}, Range=[{np.min(gray_vals)}-{np.max(gray_vals)}]")
        
        # Show histogram
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].hist(h_vals, bins=50, color='red', alpha=0.7)
        axes[0, 0].set_title('Hue Distribution')
        axes[0, 0].set_xlabel('Hue (0-180)')
        
        axes[0, 1].hist(s_vals, bins=50, color='green', alpha=0.7)
        axes[0, 1].set_title('Saturation Distribution')
        axes[0, 1].set_xlabel('Saturation (0-255)')
        
        axes[1, 0].hist(v_vals, bins=50, color='blue', alpha=0.7)
        axes[1, 0].set_title('Value/Brightness Distribution')
        axes[1, 0].set_xlabel('Value (0-255)')
        
        axes[1, 1].hist(gray_vals, bins=50, color='gray', alpha=0.7)
        axes[1, 1].set_title('Grayscale Distribution')
        axes[1, 1].set_xlabel('Intensity (0-255)')
        
        plt.tight_layout()
        plt.savefig(f'color_analysis_{os.path.basename(image_path)}.png', dpi=150)
        plt.show()
    
    def save_model(self, filepath='trichrome_model.pkl'):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='trichrome_model.pkl'):
        """Load trained model and scaler"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        print(f"Model loaded from {filepath}")
    
    # ========== EXPLAINABILITY FEATURES ==========
    
    def generate_contribution_heatmap(self, image, fibrosis_mask):
        """
        Create heatmap showing fibrosis severity
        Darker colors = higher saturation = worse fibrosis
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # Create heatmap based on saturation intensity
        heatmap = np.zeros_like(image, dtype=np.float32)
        
        # Only show heatmap where fibrosis detected
        mask_bool = fibrosis_mask > 0
        
        # Normalize saturation to 0-255 range for visualization
        if np.any(mask_bool):
            sat_normalized = np.zeros_like(saturation, dtype=np.uint8)
            sat_values = saturation[mask_bool]
            if sat_values.max() > sat_values.min():
                sat_normalized[mask_bool] = ((saturation[mask_bool] - sat_values.min()) / 
                                             (sat_values.max() - sat_values.min()) * 255).astype(np.uint8)
        
            # Apply colormap (blue to red gradient)
            heatmap_colored = cv2.applyColorMap(sat_normalized, cv2.COLORMAP_JET)
            heatmap[mask_bool] = heatmap_colored[mask_bool]
        
        # Blend with original image
        overlay = cv2.addWeighted(image.astype(np.float32), 0.6, heatmap, 0.4, 0)
        
        return overlay.astype(np.uint8)
    
    def explain_features(self, features, prediction):
        """
        Generate feature importance explanation
        Shows which features contributed most to the prediction
        """
        feature_names = [
            'Area %', 'Hue Mean', 'Hue Std', 'Sat Mean', 'Sat Std',
            'Value Mean', 'LAB_B Mean', 'LAB_B Std', 'Spread X', 
            'Spread Y', 'Num Regions', 'Avg Region Size', 
            'Region Size Std', 'Max Region Size'
        ]
        
        explanation = {
            'prediction': prediction,
            'features': {}
        }
        
        if hasattr(self.model, 'feature_importances_'):
            # Random Forest - feature importances
            importances = self.model.feature_importances_
            
            for name, feat_val, importance in zip(feature_names, features, importances):
                explanation['features'][name] = {
                    'value': float(feat_val),
                    'importance': float(importance),
                    'contribution': float(feat_val * importance)
                }
        
        elif hasattr(self.model, 'coef_'):
            # Ridge/Linear - coefficients
            coefficients = self.model.coef_
            
            for name, feat_val, coef in zip(feature_names, features, coefficients):
                explanation['features'][name] = {
                    'value': float(feat_val),
                    'coefficient': float(coef),
                    'contribution': float(feat_val * coef)
                }
        
        # Sort by absolute contribution
        sorted_features = sorted(
            explanation['features'].items(),
            key=lambda x: abs(x[1].get('contribution', 0)),
            reverse=True
        )
        
        explanation['top_features'] = [
            {'name': name, **data} 
            for name, data in sorted_features[:5]
        ]
        
        return explanation
    
    def visualize_explanation(self, image_path, save_path=None):
        """
        Create comprehensive explainability visualization
        """
        image = cv2.imread(image_path)
        if image is None:
            return
        
        max_dim = 1024
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get prediction and features
        fibrosis_mask = self.segment_fibrosis(image)
        features = self.extract_features(image, fibrosis_mask)
        
        if self.model is not None:
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            explanation = self.explain_features(features, prediction)
        else:
            prediction = (np.sum(fibrosis_mask > 0) / fibrosis_mask.size) * 100
            explanation = None
        
        # Generate heatmap
        heatmap = self.generate_contribution_heatmap(image, fibrosis_mask)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_rgb)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Segmentation mask
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(fibrosis_mask, cmap='gray')
        ax2.set_title('Fibrosis Segmentation', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Severity heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(heatmap_rgb)
        ax3.set_title('Fibrosis Severity Heatmap', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Feature importance
        if explanation:
            ax4 = fig.add_subplot(gs[1, :])
            
            top_features = explanation['top_features']
            names = [f['name'] for f in top_features]
            contributions = [f.get('contribution', 0) for f in top_features]
            
            colors = ['#48bb78' if c > 0 else '#f56565' for c in contributions]
            
            bars = ax4.barh(names, contributions, color=colors)
            ax4.set_xlabel('Contribution to Prediction', fontsize=12)
            ax4.set_title(f'Top 5 Features Contributing to {prediction:.1f}% Prediction', 
                         fontsize=14, fontweight='bold')
            ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax4.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, contrib in zip(bars, contributions):
                width = bar.get_width()
                label_x = width + (0.5 if width > 0 else -0.5)
                ax4.text(label_x, bar.get_y() + bar.get_height()/2, 
                        f'{contrib:.2f}',
                        ha='left' if width > 0 else 'right',
                        va='center', fontsize=10)
        
        plt.suptitle(f'Explainability Analysis: {os.path.basename(image_path)}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return explanation
    
    # ========== CLINICAL VALIDATION ==========
    
    def clinical_validation_study(self, images_paths, ground_truth_labels, 
                                  pathologist_scores=None):
        """
        Perform clinical validation comparing algorithm to ground truth
        
        Args:
            images_paths: List of image file paths
            ground_truth_labels: List of ground truth percentages
            pathologist_scores: Optional dict of {image_path: [path1_score, path2_score, ...]}
        """
        print("\n" + "="*70)
        print("CLINICAL VALIDATION STUDY")
        print("="*70)
        
        predictions = []
        
        for img_path in images_paths:
            pred, _ = self.predict_fibrosis(img_path)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth_labels)
        
        # Basic metrics
        mae = mean_absolute_error(ground_truth, predictions)
        r2 = r2_score(ground_truth, predictions)
        pearson_r, pearson_p = pearsonr(ground_truth, predictions)
        
        # Bias and limits of agreement (Bland-Altman)
        differences = predictions - ground_truth
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        # Categorical agreement
        def categorize(percent):
            if percent < 10: return "Minimal"
            elif percent < 25: return "Mild"
            elif percent < 50: return "Moderate"
            else: return "Severe"
        
        categories_true = [categorize(x) for x in ground_truth]
        categories_pred = [categorize(x) for x in predictions]
        
        from sklearn.metrics import cohen_kappa_score, confusion_matrix
        kappa = cohen_kappa_score(categories_true, categories_pred)
        
        # Print results
        print(f"\n{'Metric':<30} {'Value'}")
        print("-"*50)
        print(f"{'Mean Absolute Error':<30} {mae:.2f}%")
        print(f"{'R² Score':<30} {r2:.3f}")
        print(f"{'Pearson Correlation':<30} {pearson_r:.3f} (p={pearson_p:.4f})")
        print(f"{'Mean Bias':<30} {mean_diff:.2f}%")
        print(f"{'95% Limits of Agreement':<30} [{mean_diff-1.96*std_diff:.2f}, {mean_diff+1.96*std_diff:.2f}]")
        print(f"{'Cohen Kappa (Categories)':<30} {kappa:.3f}")
        
        # Confusion matrix for categories
        print("\n Confusion Matrix (Categories):")
        cm = confusion_matrix(categories_true, categories_pred, 
                             labels=["Minimal", "Mild", "Moderate", "Severe"])
        print(pd.DataFrame(cm, 
                          index=["Minimal", "Mild", "Moderate", "Severe"],
                          columns=["Minimal", "Mild", "Moderate", "Severe"]))
        
        # Inter-rater reliability if pathologist scores provided
        if pathologist_scores:
            print("\n=== INTER-RATER RELIABILITY ===")
            
            # Calculate ICC (Intraclass Correlation Coefficient)
            all_scores = []
            for img_path in images_paths:
                if img_path in pathologist_scores:
                    scores = pathologist_scores[img_path]
                    all_scores.append(scores)
            
            if len(all_scores) > 0:
                icc = self.calculate_icc(np.array(all_scores))
                print(f"Intraclass Correlation (Pathologists): {icc:.3f}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Scatter plot
        axes[0, 0].scatter(ground_truth, predictions, alpha=0.6)
        axes[0, 0].plot([0, 100], [0, 100], 'r--', label='Perfect Agreement')
        axes[0, 0].set_xlabel('Ground Truth (%)', fontsize=12)
        axes[0, 0].set_ylabel('Algorithm Prediction (%)', fontsize=12)
        axes[0, 0].set_title(f'Algorithm vs Ground Truth (R²={r2:.3f})', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Bland-Altman plot
        mean_values = (predictions + ground_truth) / 2
        axes[0, 1].scatter(mean_values, differences, alpha=0.6)
        axes[0, 1].axhline(mean_diff, color='red', linestyle='-', label=f'Mean Bias: {mean_diff:.2f}%')
        axes[0, 1].axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', 
                          label=f'95% LoA: ±{1.96*std_diff:.2f}%')
        axes[0, 1].axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Mean of Algorithm and Ground Truth (%)', fontsize=12)
        axes[0, 1].set_ylabel('Difference (Algorithm - Ground Truth) (%)', fontsize=12)
        axes[0, 1].set_title('Bland-Altman Plot', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Residuals histogram
        axes[1, 0].hist(differences, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Prediction Error (%)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Distribution of Prediction Errors', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Category confusion matrix visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = axes[1, 1].imshow(cm_normalized, cmap='Blues', aspect='auto')
        axes[1, 1].set_xticks(range(4))
        axes[1, 1].set_yticks(range(4))
        axes[1, 1].set_xticklabels(["Minimal", "Mild", "Moderate", "Severe"])
        axes[1, 1].set_yticklabels(["Minimal", "Mild", "Moderate", "Severe"])
        axes[1, 1].set_xlabel('Predicted Category', fontsize=12)
        axes[1, 1].set_ylabel('True Category', fontsize=12)
        axes[1, 1].set_title(f'Category Agreement (κ={kappa:.3f})', fontweight='bold')
        
        # Add text annotations
        for i in range(4):
            for j in range(4):
                text = axes[1, 1].text(j, i, f'{cm[i, j]}\\n({cm_normalized[i, j]:.2f})',
                                      ha="center", va="center", color="white" if cm_normalized[i, j] > 0.5 else "black")
        
        plt.colorbar(im, ax=axes[1, 1])
        plt.tight_layout()
        plt.savefig('clinical_validation_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return {
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_r,
            'bias': mean_diff,
            'kappa': kappa,
            'predictions': predictions,
            'ground_truth': ground_truth
        }
    
    def calculate_icc(self, scores_matrix):
        """
        Calculate Intraclass Correlation Coefficient (ICC)
        scores_matrix: n_images × n_raters matrix
        """
        n, k = scores_matrix.shape
        
        # Mean of all scores
        grand_mean = np.mean(scores_matrix)
        
        # Between-subjects variance
        subject_means = np.mean(scores_matrix, axis=1)
        bms = k * np.var(subject_means, ddof=1)
        
        # Within-subjects variance
        wms = np.mean(np.var(scores_matrix, axis=1, ddof=1))
        
        # ICC(2,k) - two-way random effects, average measures
        icc = (bms - wms) / (bms + (k-1)*wms)
        
        return icc
    
    # ========== PERFORMANCE OPTIMIZATION ==========
    
    def predict_optimized(self, image_path):
        """
        Optimized prediction with performance profiling
        """
        timings = {}
        total_start = time.time()
        
        # Load image
        t0 = time.time()
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        max_dim = 1024
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        timings['load_resize'] = (time.time() - t0) * 1000
        
        # Segmentation (parallelized)
        t0 = time.time()
        fibrosis_mask = self.segment_fibrosis_parallel(image)
        timings['segmentation'] = (time.time() - t0) * 1000
        
        # Feature extraction
        t0 = time.time()
        features = self.extract_features(image, fibrosis_mask)
        timings['features'] = (time.time() - t0) * 1000
        
        # ML prediction
        t0 = time.time()
        raw_percent = (np.sum(fibrosis_mask > 0) / fibrosis_mask.size) * 100
        
        if self.model is not None:
            features_scaled = self.scaler.transform([features])
            ml_prediction = self.model.predict(features_scaled)[0]
            ml_prediction = np.clip(ml_prediction, 0, 100)
        else:
            ml_prediction = None
        timings['ml_inference'] = (time.time() - t0) * 1000
        
        timings['total'] = (time.time() - total_start) * 1000
        
        return raw_percent, ml_prediction, fibrosis_mask, timings
    
    def segment_fibrosis_parallel(self, image):
        """
        Parallelized fibrosis segmentation for better performance
        """
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Run glomeruli detection in parallel
            future_red = executor.submit(self.detect_red_pink_glomeruli, image)
            future_blue = executor.submit(self.detect_blue_green_glomeruli, image)
            future_preprocess = executor.submit(self._preprocess_for_segmentation, image)
            
            red_glomeruli = future_red.result()
            blue_glomeruli = future_blue.result()
            hsv, gray, tissue_mask = future_preprocess.result()
        
        # Continue with segmentation
        lower_blue_green = np.array([75, 15, 30])
        upper_blue_green = np.array([170, 255, 255])
        all_blue_green = cv2.inRange(hsv, lower_blue_green, upper_blue_green)
        all_blue_green = cv2.bitwise_and(all_blue_green, tissue_mask)
        
        kernel_expand = np.ones((100, 100), np.uint8)
        red_glomeruli_expanded = cv2.dilate(red_glomeruli.astype(np.uint8) * 255, 
                                            kernel_expand, iterations=1)
        roi_blue_green = cv2.bitwise_and(all_blue_green, all_blue_green,
                                         mask=red_glomeruli_expanded)
        
        fibrosis_mask = cv2.bitwise_or(all_blue_green, roi_blue_green)
        fibrosis_mask[blue_glomeruli] = 0
        fibrosis_mask[red_glomeruli] = 0
        
        light_mask = gray > 190
        fibrosis_mask[light_mask] = 0
        
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((4, 4), np.uint8)
        fibrosis_mask = cv2.morphologyEx(fibrosis_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        fibrosis_mask = cv2.morphologyEx(fibrosis_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        return fibrosis_mask
    
    def _preprocess_for_segmentation(self, image):
        """Helper for parallel preprocessing"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, tissue_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        return hsv, gray, tissue_mask
    
    # ========== BATCH PROCESSING ==========
    
    def process_batch(self, folder_path, output_csv='batch_results.csv', num_workers=4):
        """
        Process multiple images in batch with parallel processing
        """
        from multiprocessing import Pool
        import glob
        
        print(f"\nProcessing batch from: {folder_path}")
        print("="*70)
        
        # Find all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
        
        # Filter for trichrome images
        trichrome_files = [f for f in image_files if os.path.basename(f).lower().startswith('tri')]
        
        print(f"Found {len(trichrome_files)} trichrome images")
        
        if len(trichrome_files) == 0:
            print("No trichrome images found")
            return None
        
        # Process in parallel
        with Pool(num_workers) as pool:
            results = pool.map(self._process_single_for_batch, trichrome_files)
        
        # Compile results
        df_data = []
        for filepath, result in zip(trichrome_files, results):
            if result:
                df_data.append({
                    'filename': os.path.basename(filepath),
                    'filepath': filepath,
                    'raw_percent': result['raw'],
                    'ml_percent': result['ml'],
                    'processing_time_ms': result['time'],
                    'ground_truth': self.extract_percentage(os.path.basename(filepath))
                })
        
        df = pd.DataFrame(df_data)
        
        # Calculate errors if ground truth available
        if 'ground_truth' in df.columns and df['ground_truth'].notna().any():
            df['error'] = abs(df['ml_percent'] - df['ground_truth'])
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")
        
        # Summary statistics
        print("\n=== BATCH PROCESSING SUMMARY ===")
        print(f"Total images processed: {len(df)}")
        print(f"Average processing time: {df['processing_time_ms'].mean():.1f}ms")
        print(f"Average raw percentage: {df['raw_percent'].mean():.1f}%")
        if 'ml_percent' in df.columns:
            print(f"Average ML prediction: {df['ml_percent'].mean():.1f}%")
        if 'error' in df.columns:
            print(f"Average prediction error: {df['error'].mean():.1f}%")
        
        return df
    
    def _process_single_for_batch(self, image_path):
        """Helper function for batch processing"""
        try:
            start = time.time()
            raw, ml, _ = self.predict_fibrosis(image_path)
            elapsed = (time.time() - start) * 1000
            
            return {
                'raw': raw,
                'ml': ml,
                'time': elapsed
            }
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None


def load_model(self, filepath='trichrome_model.pkl'):
        """Load trained model and scaler"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    analyzer = TrichromeAnalyzer("C:/Users/Bmark/Desktop/ywa")
    
    print("Scanning for trichrome images...")
    image_files = analyzer.find_trichrome_images()
    
    # First, analyze color distribution of sample images
    print("\n" + "="*70)
    print("STEP 1: Analyzing color distribution to calibrate thresholds")
    print("="*70)
    
    # Find diverse samples: low, medium, and high fibrosis
    samples_to_analyze = []
    low_fib = []
    mid_fib = []
    high_fib = []
    
    for img_path in image_files:
        pct = analyzer.extract_percentage(os.path.basename(img_path))
        if pct is not None:
            if pct <= 15:
                low_fib.append((img_path, pct))
            elif 25 <= pct <= 35:
                mid_fib.append((img_path, pct))
            elif pct >= 60:
                high_fib.append((img_path, pct))
    
    # Pick one from each category
    if low_fib:
        samples_to_analyze.append(low_fib[0])
    if mid_fib:
        samples_to_analyze.append(mid_fib[0])
    if high_fib:
        samples_to_analyze.append(high_fib[0])
    
    print(f"\nAnalyzing {len(samples_to_analyze)} representative images...")
    for img_path, pct in samples_to_analyze:
        print(f"\n>>> Analyzing: {os.path.basename(img_path)} (Labeled: {pct}%)")
        analyzer.analyze_color_distribution(img_path)
    
    # Now proceed with training
    print("\n" + "="*70)
    print("STEP 2: Preparing training data and building model")
    print("="*70)
    
    X, y, image_paths = analyzer.prepare_training_data()
    
    print(f"\nFound {len(X)} valid training images")
    
    if len(X) > 0:
        print("\nTraining model...")
        analyzer.train_model(X, y)
        
        analyzer.save_model('trichrome_fibrosis_model.pkl')
        
        print("\nGenerating visualizations...")
        for i, img_path in enumerate(image_paths[:min(3, len(image_paths))]):
            analyzer.visualize_analysis(img_path, f'analysis_result_{i+1}.png')
        
        print("\n=== Testing on all images ===")
        for img_path in image_paths:
            predicted_pct, _ = analyzer.predict_fibrosis(img_path)
            actual_pct = analyzer.extract_percentage(os.path.basename(img_path))
            error = abs(predicted_pct - actual_pct)
            print(f"{os.path.basename(img_path):30s} | Actual: {actual_pct:5.1f}% | Predicted: {predicted_pct:5.1f}% | Error: {error:5.1f}%")
    else:
        print("No valid training images found.")
