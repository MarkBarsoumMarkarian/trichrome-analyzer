"""
Diagnostic script to understand why model performance is poor
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from trichrome_core import TrichromeAnalyzer
import os
import re

def diagnose_training_data(training_folder, n_samples=10):
    """
    Analyze training data to find issues
    """
    analyzer = TrichromeAnalyzer()
    
    # Find images
    image_files = analyzer._find_trichrome_images(training_folder)
    print(f"\nFound {len(image_files)} trichrome images")
    
    # Analyze a sample
    results = []
    
    for i, img_path in enumerate(image_files[:n_samples]):
        filename = os.path.basename(img_path)
        label = analyzer._extract_percentage(filename)
        
        if label is None:
            print(f"⚠️ No label found in: {filename}")
            continue
        
        # Load and process
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Could not load: {filename}")
            continue
        
        image = analyzer._resize_if_needed(image)
        
        # Get raw segmentation
        fibrosis_mask = analyzer.segment_fibrosis(image)
        raw_percent = (np.sum(fibrosis_mask > 0) / fibrosis_mask.size) * 100
        
        # Calculate error
        error = raw_percent - label
        
        results.append({
            'filename': filename,
            'label': label,
            'raw': raw_percent,
            'error': error
        })
        
        print(f"\n{i+1}. {filename}")
        print(f"   Label:  {label:.1f}%")
        print(f"   Raw:    {raw_percent:.1f}%")
        print(f"   Error:  {error:+.1f}%")
    
    # Summary statistics
    if results:
        errors = [r['error'] for r in results]
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Mean Error:   {np.mean(errors):+.1f}%")
        print(f"Std Error:    {np.std(errors):.1f}%")
        print(f"Mean Abs Err: {np.mean(np.abs(errors)):.1f}%")
        print(f"Min Error:    {np.min(errors):+.1f}%")
        print(f"Max Error:    {np.max(errors):+.1f}%")
        
        # Check for systematic bias
        if abs(np.mean(errors)) > 5:
            if np.mean(errors) > 0:
                print("\n⚠️ SYSTEMATIC OVERESTIMATION: Segmentation detects MORE than labels")
                print("   → Segmentation might be too aggressive")
            else:
                print("\n⚠️ SYSTEMATIC UNDERESTIMATION: Segmentation detects LESS than labels")
                print("   → Segmentation might be too conservative")
        
        # Check for high variance
        if np.std(errors) > 15:
            print("\n⚠️ HIGH VARIANCE: Inconsistent segmentation across images")
            print("   → Labels might be inconsistent OR segmentation needs tuning")
    
    return results


def visualize_worst_cases(training_folder, n_cases=5):
    """
    Show the worst predictions to understand failures
    """
    analyzer = TrichromeAnalyzer()
    image_files = analyzer._find_trichrome_images(training_folder)
    
    all_results = []
    
    print("\nAnalyzing all images to find worst cases...")
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
        
        all_results.append({
            'path': img_path,
            'filename': filename,
            'label': label,
            'raw': raw_percent,
            'abs_error': abs(raw_percent - label)
        })
    
    # Sort by worst error
    all_results.sort(key=lambda x: x['abs_error'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"TOP {n_cases} WORST PREDICTIONS:")
    print(f"{'='*80}")
    
    for i, result in enumerate(all_results[:n_cases]):
        error = result['raw'] - result['label']
        print(f"\n{i+1}. {result['filename']}")
        print(f"   Label: {result['label']:.1f}%  |  Raw: {result['raw']:.1f}%  |  Error: {error:+.1f}%")
        print(f"   Path: {result['path']}")
    
    return all_results


def check_label_distribution(training_folder):
    """
    Check if labels are evenly distributed
    """
    analyzer = TrichromeAnalyzer()
    image_files = analyzer._find_trichrome_images(training_folder)
    
    labels = []
    for img_path in image_files:
        label = analyzer._extract_percentage(os.path.basename(img_path))
        if label is not None:
            labels.append(label)
    
    print(f"\n{'='*60}")
    print("LABEL DISTRIBUTION")
    print(f"{'='*60}")
    print(f"Total labeled images: {len(labels)}")
    print(f"Min label:  {min(labels):.1f}%")
    print(f"Max label:  {max(labels):.1f}%")
    print(f"Mean label: {np.mean(labels):.1f}%")
    print(f"Median:     {np.median(labels):.1f}%")
    print(f"Std dev:    {np.std(labels):.1f}%")
    
    # Histogram
    print("\nLabel ranges:")
    bins = [0, 10, 25, 50, 100]
    bin_names = ['Minimal (<10%)', 'Mild (10-25%)', 'Moderate (25-50%)', 'Severe (>50%)']
    
    for i in range(len(bins)-1):
        count = sum(1 for l in labels if bins[i] <= l < bins[i+1])
        print(f"  {bin_names[i]}: {count} images ({count/len(labels)*100:.1f}%)")
    
    return labels


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnostic_script.py <training_folder>")
        sys.exit(1)
    
    folder = sys.argv[1]
    
    print("TRICHROME TRAINING DIAGNOSTICS")
    print("="*80)
    
    # 1. Check label distribution
    labels = check_label_distribution(folder)
    
    # 2. Diagnose sample of images
    print("\n" + "="*80)
    results = diagnose_training_data(folder, n_samples=20)
    
    # 3. Show worst cases
    print("\n" + "="*80)
    worst = visualize_worst_cases(folder, n_cases=10)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    if results:
        mean_error = np.mean([r['error'] for r in results])
        
        if abs(mean_error) > 10:
            print("1. ⚠️ Large systematic bias detected!")
            print("   → Check if labels match what segmentation should detect")
            print("   → Labels might be measuring something different")
        
        if len(set([r['label'] for r in results])) < 5:
            print("2. ⚠️ Very few unique labels!")
            print("   → Need more diverse training data")
        
        if max(labels) - min(labels) < 30:
            print("3. ⚠️ Narrow label range!")
            print("   → Model can't learn full spectrum of fibrosis")
    
    print("\n✅ Diagnostics complete!")
