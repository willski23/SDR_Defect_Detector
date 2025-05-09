import argparse
import numpy as np
import h5py
import os
import pickle
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import pandas as pd
import seaborn as sns

from detect_image import detect_defects_in_image, load_model

def calculate_metrics(predicted_defects, true_defects):
    """Calculate precision, recall, F1"""
    pred_positions = [pos for pos, _ in predicted_defects]
    
    true_positives = len(set(pred_positions) & set(true_defects))
    false_positives = len(set(pred_positions) - set(true_defects))
    false_negatives = len(set(true_defects) - set(pred_positions))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_parameter_set(model, images, true_defects, threshold, precision_mode, window_size=8, overlap=0.5, sample_size=100):
    """Evaluate a set of parameters on a sample of images"""
    # Sample images
    num_images = len(images)
    if num_images > sample_size:
        indices = np.random.choice(num_images, sample_size, replace=False)
    else:
        indices = range(num_images)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for idx in indices:
        image = images[idx]
        defects = true_defects[idx]
        
        # Detect defects with current parameters
        defect_positions = detect_defects_in_image(
            image, model, threshold, 
            window_size=window_size, 
            overlap=overlap,
            precision_mode=precision_mode
        )
        
        # Calculate metrics
        metrics = calculate_metrics(defect_positions, defects)
        
        # Update totals
        total_tp += metrics['true_positives']
        total_fp += metrics['false_positives']
        total_fn += metrics['false_negatives']
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'threshold': threshold,
        'precision_mode': precision_mode,
        'window_size': window_size,
        'overlap': overlap,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    parser = argparse.ArgumentParser(description='Optimize Detection Parameters for Ultrasound Transducer Defects')
    parser.add_argument('--model', type=str, default='defect_detection_model.pkl', help='Model file path')
    parser.add_argument('--data_file', type=str, required=True, help='HDF5 data file containing images')
    parser.add_argument('--output_dir', type=str, default='parameter_optimization', help='Directory to save output')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of images to use for optimization')
    parser.add_argument('--precision_weight', type=float, default=0.7, 
                       help='Weight for precision in optimization (0.5-1.0, higher values prefer precision over recall)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, base_threshold, model_data = load_model(args.model)
    print(f"Base threshold from model: {base_threshold:.4f}")
    
    # Load data
    print(f"Loading data from {args.data_file}")
    with h5py.File(args.data_file, 'r') as f:
        if 'images' not in f or 'dead_elements' not in f:
            print("Error: Dataset must contain 'images' and 'dead_elements'")
            return
            
        # Load all images
        images = f['images'][:]
        
        # Load all defect positions
        defect_positions = []
        for elements in f['dead_elements'][:]:
            positions = np.where(elements > 0)[0].tolist()
            defect_positions.append(positions)
    
    print(f"Loaded {len(images)} images")
    
    # Define parameter grid
    thresholds = [base_threshold * factor for factor in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]]
    precision_modes = [False, True]
    window_sizes = [8]  # Fixed for efficiency
    overlaps = [0.5]    # Fixed for efficiency
    
    # Evaluate all parameter combinations
    results = []
    
    print("Evaluating parameter combinations...")
    param_combinations = list(product(thresholds, precision_modes, window_sizes, overlaps))
    
    for threshold, precision_mode, window_size, overlap in tqdm(param_combinations):
        result = evaluate_parameter_set(
            model, images, defect_positions, 
            threshold, precision_mode, 
            window_size, overlap,
            sample_size=args.num_samples
        )
        results.append(result)
    
    # Find best parameter sets for different optimization criteria
    # 1. Best F1 score
    best_f1_result = max(results, key=lambda x: x['f1'])
    
    # 2. Best precision (with at least 0.4 recall)
    valid_precision_results = [r for r in results if r['recall'] >= 0.4]
    best_precision_result = max(valid_precision_results, key=lambda x: x['precision']) if valid_precision_results else None
    
    # 3. Best recall (with at least 0.5 precision)
    valid_recall_results = [r for r in results if r['precision'] >= 0.5]
    best_recall_result = max(valid_recall_results, key=lambda x: x['recall']) if valid_recall_results else None
    
    # 4. Custom weighted score based on precision_weight
    for r in results:
        r['weighted_score'] = (args.precision_weight * r['precision'] + 
                              (1 - args.precision_weight) * r['recall'])
    
    best_weighted_result = max(results, key=lambda x: x['weighted_score'])
    
    # Print best results
    print("\nBest Parameters:")
    print("\n1. Best F1 Score:")
    print(f"   Threshold: {best_f1_result['threshold']:.4f}")
    print(f"   Precision Mode: {best_f1_result['precision_mode']}")
    print(f"   Metrics - Precision: {best_f1_result['precision']:.4f}, Recall: {best_f1_result['recall']:.4f}, F1: {best_f1_result['f1']:.4f}")
    
    if best_precision_result:
        print("\n2. Best Precision (with recall >= 0.4):")
        print(f"   Threshold: {best_precision_result['threshold']:.4f}")
        print(f"   Precision Mode: {best_precision_result['precision_mode']}")
        print(f"   Metrics - Precision: {best_precision_result['precision']:.4f}, Recall: {best_precision_result['recall']:.4f}, F1: {best_precision_result['f1']:.4f}")
    
    if best_recall_result:
        print("\n3. Best Recall (with precision >= 0.5):")
        print(f"   Threshold: {best_recall_result['threshold']:.4f}")
        print(f"   Precision Mode: {best_recall_result['precision_mode']}")
        print(f"   Metrics - Precision: {best_recall_result['precision']:.4f}, Recall: {best_recall_result['recall']:.4f}, F1: {best_recall_result['f1']:.4f}")
    
    print(f"\n4. Best Weighted Score (precision_weight={args.precision_weight}):")
    print(f"   Threshold: {best_weighted_result['threshold']:.4f}")
    print(f"   Precision Mode: {best_weighted_result['precision_mode']}")
    print(f"   Metrics - Precision: {best_weighted_result['precision']:.4f}, Recall: {best_weighted_result['recall']:.4f}, F1: {best_weighted_result['f1']:.4f}")
    
    # Create visualization of parameter space
    df = pd.DataFrame(results)
    
    # Create parameter combinations as strings for better visualization
    df['param_combo'] = df.apply(lambda row: f"T={row['threshold']:.2f}, PM={row['precision_mode']}", axis=1)
    
    # Precision-Recall plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(df['recall'], df['precision'], c=df['f1'], cmap='viridis', 
                        s=100, alpha=0.7)
    
    # Add labels for best points
    plt.scatter(best_f1_result['recall'], best_f1_result['precision'], 
               s=200, facecolors='none', edgecolors='red', linewidth=2, label='Best F1')
    
    if best_precision_result:
        plt.scatter(best_precision_result['recall'], best_precision_result['precision'], 
                   s=200, facecolors='none', edgecolors='blue', linewidth=2, label='Best Precision')
    
    if best_recall_result:
        plt.scatter(best_recall_result['recall'], best_recall_result['precision'], 
                   s=200, facecolors='none', edgecolors='green', linewidth=2, label='Best Recall')
    
    plt.scatter(best_weighted_result['recall'], best_weighted_result['precision'], 
               s=200, facecolors='none', edgecolors='purple', linewidth=2, label='Best Weighted')
    
    plt.colorbar(scatter, label='F1 Score')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall for Different Parameter Combinations')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'parameter_precision_recall.png'))
    
    # Threshold vs metrics plot
    plt.figure(figsize=(14, 8))
    
    # Group by threshold and precision_mode
    grouped = df.groupby(['threshold', 'precision_mode']).mean().reset_index()
    
    # Create separate plots for each precision_mode
    for mode in [False, True]:
        mode_data = grouped[grouped['precision_mode'] == mode]
        plt.plot(mode_data['threshold'], mode_data['precision'], 
                marker='o', label=f'Precision (PM={mode})')
        plt.plot(mode_data['threshold'], mode_data['recall'], 
                marker='s', label=f'Recall (PM={mode})')
        plt.plot(mode_data['threshold'], mode_data['f1'], 
                marker='^', label=f'F1 (PM={mode})')
    
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs. Threshold for Different Precision Modes')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'threshold_vs_metrics.png'))
    
    # Save results to CSV for further analysis
    df.to_csv(os.path.join(args.output_dir, 'parameter_optimization_results.csv'), index=False)
    
    # Save best parameters as JSON
    best_params = {
        'best_f1': best_f1_result,
        'best_precision': best_precision_result if best_precision_result else None,
        'best_recall': best_recall_result if best_recall_result else None,
        'best_weighted': best_weighted_result,
        'optimization_settings': {
            'num_samples': args.num_samples,
            'precision_weight': args.precision_weight,
            'base_threshold': float(base_threshold)
        }
    }
    
    with open(os.path.join(args.output_dir, 'best_parameters.json'), 'w') as f:
        json.dump(best_params, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Run batch processing with these optimal parameters:")
    print(f"python batch_process.py --model {args.model} --data_file {args.data_file} --output_dir results")
    print(f"    --threshold {best_weighted_result['threshold']:.4f} {'--precision_mode' if best_weighted_result['precision_mode'] else ''}")

if __name__ == "__main__":
    main()