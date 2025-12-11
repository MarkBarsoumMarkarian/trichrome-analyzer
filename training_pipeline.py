"""
Complete Training Pipeline
===========================

End-to-end pipeline incorporating all advanced features:
1. Color normalization
2. U-Net training
3. Ensemble creation
4. Active learning

Usage:
    python training_pipeline.py --mode all --data-dir /path/to/data
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from advanced_features import (
    MacenkoNormalizer, 
    UNet, 
    TrichromeDataset,
    train_unet,
    predict_with_unet,
    EnsemblePredictor,
    ActiveLearningPipeline
)
from trichrome_analyzer import TrichromeAnalyzer

# ==================== STEP 1: COLOR NORMALIZATION ====================

def setup_color_normalization(reference_image_path, output_path='macenko_normalizer.pkl'):
    """
    Setup and save color normalizer
    
    Args:
        reference_image_path: Path to reference trichrome image with ideal staining
        output_path: Where to save the fitted normalizer
    """
    print("\n" + "="*70)
    print("STEP 1: COLOR NORMALIZATION SETUP")
    print("="*70)
    
    normalizer = MacenkoNormalizer()
    
    # Load reference image
    ref_image = cv2.imread(reference_image_path)
    if ref_image is None:
        print(f"✗ Error: Could not load reference image: {reference_image_path}")
        return None
    
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    
    print(f"✓ Loaded reference image: {reference_image_path}")
    print(f"  Shape: {ref_image.shape}")
    
    # Fit normalizer
    print("  Fitting Macenko normalizer...")
    normalizer.fit(ref_image)
    
    # Save
    normalizer.save(output_path)
    print(f"✓ Normalizer saved to: {output_path}")
    
    return normalizer


def normalize_dataset(normalizer, input_folder, output_folder):
    """
    Normalize all images in a folder
    
    Args:
        normalizer: Fitted MacenkoNormalizer
        input_folder: Folder with original images
        output_folder: Where to save normalized images
    """
    print("\nNormalizing dataset...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).rglob(ext))
    
    print(f"Found {len(image_files)} images to normalize")
    
    for i, img_path in enumerate(image_files):
        # Load
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = normalizer.transform(image)
        
        # Save
        relative_path = img_path.relative_to(input_folder)
        output_path = Path(output_folder) / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        normalized_bgr = cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), normalized_bgr)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(image_files)} images")
    
    print(f"✓ All images normalized and saved to: {output_folder}")


# ==================== STEP 2: U-NET TRAINING ====================

def prepare_unet_data(data_folder, mask_folder, val_split=0.2):
    """
    Prepare data for U-Net training
    
    Args:
        data_folder: Folder with training images
        mask_folder: Folder with corresponding segmentation masks
        val_split: Validation split ratio
    
    Returns:
        train_loader, val_loader
    """
    print("\n" + "="*70)
    print("STEP 2: U-NET DATA PREPARATION")
    print("="*70)
    
    # Find matching image-mask pairs
    image_files = sorted(Path(data_folder).rglob('*.jpg'))
    mask_files = []
    
    for img_path in image_files:
        mask_path = Path(mask_folder) / img_path.relative_to(data_folder)
        mask_path = mask_path.with_suffix('.png')
        
        if mask_path.exists():
            mask_files.append(str(mask_path))
        else:
            print(f"⚠ Warning: No mask found for {img_path}")
    
    image_paths = [str(p) for p in image_files[:len(mask_files)]]
    
    print(f"✓ Found {len(image_paths)} image-mask pairs")
    
    # Split train/val
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_files, test_size=val_split, random_state=42
    )
    
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    
    # Create datasets
    train_dataset = TrichromeDataset(train_images, train_masks, augment=True)
    val_dataset = TrichromeDataset(val_images, val_masks, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


def train_unet_model(train_loader, val_loader, num_epochs=50, device=None):
    """Train U-Net model"""
    print("\n" + "="*70)
    print("STEP 2: U-NET TRAINING")
    print("="*70)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    model, history = train_unet(train_loader, val_loader, num_epochs, device)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('U-Net Training History')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('unet_training_history.png', dpi=150, bbox_inches='tight')
    print("✓ Training history saved to: unet_training_history.png")
    plt.show()
    
    print("\n✓ U-Net training complete!")
    print("  Best model saved to: unet_best.pth")
    
    return model


# ==================== STEP 3: ENSEMBLE CREATION ====================

def create_ensemble(traditional_model_path, unet_model_path, val_images, val_labels):
    """
    Create and optimize ensemble
    
    Args:
        traditional_model_path: Path to traditional model pickle
        unet_model_path: Path to U-Net weights
        val_images: List of validation image paths
        val_labels: List of ground truth percentages
    """
    print("\n" + "="*70)
    print("STEP 3: ENSEMBLE CREATION")
    print("="*70)
    
    ensemble = EnsemblePredictor()
    
    # 1. Load traditional analyzer
    print("Loading traditional analyzer...")
    traditional_analyzer = TrichromeAnalyzer(".")
    traditional_analyzer.load_model(traditional_model_path)
    ensemble.add_model(traditional_analyzer, 'traditional')
    print("✓ Traditional model added")
    
    # 2. Load U-Net
    print("Loading U-Net...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet = UNet(in_channels=3, out_channels=3).to(device)
    unet.load_state_dict(torch.load(unet_model_path, map_location=device))
    ensemble.add_model(unet, 'unet')
    print("✓ U-Net model added")
    
    # 3. Fit ensemble weights
    print("\nOptimizing ensemble weights on validation set...")
    print(f"  Using {len(val_images)} validation images")
    
    # Load validation images
    X_val = [cv2.imread(path) for path in val_images]
    y_val = np.array(val_labels)
    
    ensemble.fit_weights(X_val, y_val)
    
    # 4. Evaluate ensemble
    print("\n=== ENSEMBLE EVALUATION ===")
    
    ensemble_preds = []
    individual_preds = {f'model_{i}': [] for i in range(len(ensemble.models))}
    
    for image in X_val:
        ens_pred, ind_preds = ensemble.predict(image)
        ensemble_preds.append(ens_pred)
        
        for i, pred in enumerate(ind_preds):
            individual_preds[f'model_{i}'].append(pred)
    
    ensemble_preds = np.array(ensemble_preds)
    
    from sklearn.metrics import mean_absolute_error, r2_score
    
    print(f"\nEnsemble MAE: {mean_absolute_error(y_val, ensemble_preds):.2f}%")
    print(f"Ensemble R²: {r2_score(y_val, ensemble_preds):.3f}")
    
    for i, (model_name, preds) in enumerate(individual_preds.items()):
        mae = mean_absolute_error(y_val, preds)
        print(f"  {ensemble.models[i]['type']} MAE: {mae:.2f}%")
    
    # Save ensemble
    ensemble.save('ensemble_config.json')
    print("\n✓ Ensemble configuration saved to: ensemble_config.json")
    
    return ensemble


# ==================== STEP 4: ACTIVE LEARNING ====================

def setup_active_learning(ensemble, unlabeled_folder, n_samples=10):
    """
    Setup active learning pipeline
    
    Args:
        ensemble: Trained ensemble model
        unlabeled_folder: Folder with unlabeled images
        n_samples: Number of samples to select
    """
    print("\n" + "="*70)
    print("STEP 4: ACTIVE LEARNING SETUP")
    print("="*70)
    
    active_learner = ActiveLearningPipeline(ensemble)
    
    # Find unlabeled images
    unlabeled_images = list(Path(unlabeled_folder).rglob('*.jpg'))
    unlabeled_paths = [str(p) for p in unlabeled_images]
    
    print(f"✓ Found {len(unlabeled_paths)} unlabeled images")
    
    # Add to pool
    active_learner.add_unlabeled_data(unlabeled_paths)
    
    # Select samples
    print(f"\nSelecting {n_samples} most informative samples for labeling...")
    selected = active_learner.select_samples_for_labeling(n_samples, strategy='hybrid')
    
    print("\n=== SELECTED SAMPLES FOR LABELING ===")
    for i, path in enumerate(selected, 1):
        print(f"{i}. {Path(path).name}")
    
    # Generate report
    active_learner.generate_report('active_learning_report.png')
    
    # Save selected samples to file
    with open('samples_to_label.txt', 'w') as f:
        for path in selected:
            f.write(f"{path}\n")
    
    print("\n✓ Selected samples saved to: samples_to_label.txt")
    print("  Please have pathologist label these images and update labels.csv")
    
    return active_learner


# ==================== MAIN PIPELINE ====================

def run_complete_pipeline(args):
    """Run the complete training pipeline"""
    
    print("\n" + "="*70)
    print("TRICHROME ANALYZER - ADVANCED TRAINING PIPELINE")
    print("="*70)
    print(f"\nData directory: {args.data_dir}")
    print(f"Mode: {args.mode}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Step 1: Color Normalization
    if args.mode in ['all', 'normalize']:
        if args.reference_image:
            normalizer = setup_color_normalization(
                args.reference_image,
                'macenko_normalizer.pkl'
            )
            
            if normalizer and args.normalize_data:
                normalize_dataset(
                    normalizer,
                    args.data_dir,
                    args.data_dir + '_normalized'
                )
        else:
            print("⚠ Skipping normalization: --reference-image not provided")
    
    # Step 2: U-Net Training
    if args.mode in ['all', 'unet']:
        if args.mask_dir:
            train_loader, val_loader = prepare_unet_data(
                args.data_dir,
                args.mask_dir,
                val_split=0.2
            )
            
            unet_model = train_unet_model(
                train_loader,
                val_loader,
                num_epochs=args.unet_epochs
            )
        else:
            print("⚠ Skipping U-Net training: --mask-dir not provided")
    
    # Step 3: Ensemble
    if args.mode in ['all', 'ensemble']:
        if args.traditional_model and args.unet_model and args.val_csv:
            import pandas as pd
            
            val_df = pd.read_csv(args.val_csv)
            val_images = val_df['filepath'].tolist()
            val_labels = val_df['ground_truth'].tolist()
            
            ensemble = create_ensemble(
                args.traditional_model,
                args.unet_model,
                val_images,
                val_labels
            )
        else:
            print("⚠ Skipping ensemble: Missing required files")
    
    # Step 4: Active Learning
    if args.mode in ['all', 'active']:
        if args.unlabeled_dir and 'ensemble' in locals():
            active_learner = setup_active_learning(
                ensemble,
                args.unlabeled_dir,
                n_samples=args.n_active_samples
            )
        else:
            print("⚠ Skipping active learning: --unlabeled-dir not provided or ensemble not created")
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Advanced Training Pipeline for Trichrome Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['all', 'normalize', 'unet', 'ensemble', 'active'],
                       help='Which steps to run')
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training images')
    
    # Color normalization
    parser.add_argument('--reference-image', type=str,
                       help='Reference image for color normalization')
    parser.add_argument('--normalize-data', action='store_true',
                       help='Normalize all images in data-dir')
    
    # U-Net training
    parser.add_argument('--mask-dir', type=str,
                       help='Directory containing segmentation masks')
    parser.add_argument('--unet-epochs', type=int, default=50,
                       help='Number of U-Net training epochs')
    
    # Ensemble
    parser.add_argument('--traditional-model', type=str,
                       help='Path to traditional model (.pkl)')
    parser.add_argument('--unet-model', type=str,
                       help='Path to U-Net weights (.pth)')
    parser.add_argument('--val-csv', type=str,
                       help='CSV with validation data (filepath, ground_truth)')
    
    # Active learning
    parser.add_argument('--unlabeled-dir', type=str,
                       help='Directory with unlabeled images')
    parser.add_argument('--n-active-samples', type=int, default=10,
                       help='Number of samples to select for labeling')
    
    args = parser.parse_args()
    
    run_complete_pipeline(args)


if __name__ == "__main__":
    main()
