"""
Visualize what the segmentation is actually detecting
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from trichrome_core import TrichromeAnalyzer
import os

def visualize_segmentation(image_path, save_path=None):
    """
    Show original image, what's being detected, and overlay
    """
    analyzer = TrichromeAnalyzer()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {image_path}")
        return
    
    image = analyzer._resize_if_needed(image)
    
    # Get segmentation
    fibrosis_mask = analyzer.segment_fibrosis(image)
    raw_percent = (np.sum(fibrosis_mask > 0) / fibrosis_mask.size) * 100
    
    # Get individual components for debugging
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Blue/green mask (what should be fibrosis)
    lower_blue_green = np.array([75, 15, 30])
    upper_blue_green = np.array([170, 255, 255])
    blue_green_mask = cv2.inRange(hsv, lower_blue_green, upper_blue_green)
    
    # Red/pink areas (should be excluded)
    red_glom = analyzer.detect_red_pink_glomeruli(image)
    blue_glom = analyzer.detect_blue_green_glomeruli(image)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Blue/green detection
    axes[0, 1].imshow(blue_green_mask, cmap='gray')
    axes[0, 1].set_title('Blue/Green Detection\n(Raw color mask)')
    axes[0, 1].axis('off')
    
    # Final fibrosis mask
    axes[0, 2].imshow(fibrosis_mask, cmap='hot')
    axes[0, 2].set_title(f'Final Fibrosis Mask\nDetected: {raw_percent:.1f}%')
    axes[0, 2].axis('off')
    
    # Red glomeruli
    axes[1, 0].imshow(red_glom, cmap='Reds')
    axes[1, 0].set_title('Red Glomeruli (excluded)')
    axes[1, 0].axis('off')
    
    # Blue glomeruli
    axes[1, 1].imshow(blue_glom, cmap='Blues')
    axes[1, 1].set_title('Blue Glomeruli (excluded)')
    axes[1, 1].axis('off')
    
    # Overlay
    overlay = image.copy()
    overlay[fibrosis_mask > 0] = [0, 255, 255]  # Cyan for detected fibrosis
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    axes[1, 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Overlay (Cyan = Detected)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return raw_percent


def batch_visualize_worst(training_folder, output_folder='debug_visualizations', n_cases=10):
    """
    Visualize the worst segmentation cases
    """
    os.makedirs(output_folder, exist_ok=True)
    
    analyzer = TrichromeAnalyzer()
    image_files = analyzer._find_trichrome_images(training_folder)
    
    results = []
    
    print("Finding worst cases...")
    for img_path in image_files:
        filename = os.path.basename(img_path)
        label = analyzer._extract_percentage(filename)
        
        if label is None:
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        image = analyzer._resize_if_needed(image)
        fibrosis_mask = analyzer.segment_fibrosis(image)
        raw_percent = (np.sum(fibrosis_mask > 0) / fibrosis_mask.size) * 100
        
        results.append({
            'path': img_path,
            'filename': filename,
            'label': label,
            'raw': raw_percent,
            'abs_error': abs(raw_percent - label)
        })
    
    # Sort by worst error
    results.sort(key=lambda x: x['abs_error'], reverse=True)
    
    print(f"\nGenerating visualizations for top {n_cases} worst cases...")
    
    for i, result in enumerate(results[:n_cases]):
        print(f"{i+1}. {result['filename']} - Label: {result['label']:.1f}% | Raw: {result['raw']:.1f}%")
        
        output_path = os.path.join(output_folder, f"debug_{i+1:02d}_{result['filename']}")
        visualize_segmentation(result['path'], output_path)
    
    print(f"\n✅ Visualizations saved to '{output_folder}/' folder")


def test_hsv_ranges(image_path):
    """
    Interactively test different HSV ranges to find what works
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {image_path}")
        return
    
    analyzer = TrichromeAnalyzer()
    image = analyzer._resize_if_needed(image)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Test various ranges
    test_ranges = [
        ("Current (75-170)", [75, 15, 30], [170, 255, 255]),
        ("Blue only (90-130)", [90, 30, 30], [130, 255, 255]),
        ("Cyan (80-100)", [80, 30, 30], [100, 255, 255]),
        ("Green-Blue (85-125)", [85, 40, 40], [125, 255, 255]),
        ("Strict Blue (100-120)", [100, 50, 50], [120, 255, 255]),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Show original
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Test each range
    for i, (name, lower, upper) in enumerate(test_ranges, 1):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        percent = (np.sum(mask > 0) / mask.size) * 100
        
        axes[i].imshow(mask, cmap='hot')
        axes[i].set_title(f'{name}\nDetected: {percent:.1f}%')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python segmentation_visualizer.py <training_folder>")
        print("  python segmentation_visualizer.py single <image_path>")
        print("  python segmentation_visualizer.py test <image_path>")
        sys.exit(1)
    
    if sys.argv[1] == 'single' and len(sys.argv) > 2:
        # Visualize single image
        visualize_segmentation(sys.argv[2])
    elif sys.argv[1] == 'test' and len(sys.argv) > 2:
        # Test HSV ranges
        test_hsv_ranges(sys.argv[2])
    else:
        # Batch process worst cases
        batch_visualize_worst(sys.argv[1])
