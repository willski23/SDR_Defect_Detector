import argparse
import numpy as np
import h5py
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_models(window_model_path, element_model_path=None):
    """Load both window-level and element-level models"""
    import pickle
    
    # Load window model
    print(f"Loading window model from {window_model_path}")
    with open(window_model_path, 'rb') as f:
        window_model_data = pickle.load(f)
    
    window_model = window_model_data['model']
    window_threshold = window_model_data['threshold']
    
    # Load element model if provided
    element_model = None
    element_threshold = None
    
    if element_model_path:
        print(f"Loading element model from {element_model_path}")
        try:
            with open(element_model_path, 'rb') as f:
                element_model_data = pickle.load(f)
            
            element_model = element_model_data['model']
            element_threshold = element_model_data['threshold']
        except Exception as e:
            print(f"Error loading element model: {e}")
    
    return window_model, window_threshold, element_model, element_threshold

def calculate_metrics(predicted_defects, true_defects):
    """Calculate precision, recall, F1"""
    pred_positions = [pos for pos, _ in predicted_defects]
    
    # Convert to sets for more efficient intersection operations
    pred_set = set(pred_positions)
    true_set = set(true_defects)
    
    true_positives = len(pred_set & true_set)
    false_positives = len(pred_set - true_set)
    false_negatives = len(true_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate F-beta (prioritize precision)
    beta = 0.5  # Strong precision focus
    if precision + recall > 0:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    else:
        f_beta = 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f_beta': f_beta
    }

def evaluate_thresholds(window_model_path, element_model_path, data_file, output_dir):
    """Evaluate model performance across different threshold combinations"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Import detection functions
    from enhanced_detection import enhanced_detection
    
    # Load models
    window_model, base_window_threshold, element_model, base_element_threshold = load_models(
        window_model_path, element_model_path
    )
    
    # Load validation data
    with h5py.File(data_file, 'r') as f:
        # Use a subset for evaluation
        images = f['images'][:200]  # First 200 images
        dead_elements = f['dead_elements'][:200]
        
        # Convert to ground truth format
        true_defects = []
        for elements in dead_elements:
            positions = np.where(elements > 0)[0].tolist()
            true_defects.append(positions)
    
    # Define threshold combinations to evaluate
    window_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    element_factors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Store results for each combination
    results = []
    
    # Track best thresholds for different metrics
    best_f1 = 0
    best_f1_thresholds = None
    best_precision = 0
    best_precision_thresholds = None
    best_fbeta = 0
    best_fbeta_thresholds = None
    
    # Evaluate each combination
    for w_factor in window_factors:
        for e_factor in element_factors:
            window_threshold = base_window_threshold * w_factor
            element_threshold = base_element_threshold * e_factor
            
            # Track metrics
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            # Process images
            for i, image in enumerate(tqdm(images, desc=f"W:{w_factor:.1f} E:{e_factor:.1f}")):
                # Run detection
                defect_positions = enhanced_detection(
                    image, window_model, window_threshold,
                    element_model, element_threshold
                )
                
                # Calculate metrics
                metrics = calculate_metrics(defect_positions, true_defects[i])
                
                # Update totals
                total_tp += metrics['true_positives']
                total_fp += metrics['false_positives']
                total_fn += metrics['false_negatives']
            
            # Calculate overall metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate F-beta (beta=0.5 prioritizes precision)
            beta = 0.5
            f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
            
            # Save result
            result = {
                'window_factor': w_factor,
                'element_factor': e_factor,
                'window_threshold': window_threshold,
                'element_threshold': element_threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'f_beta': f_beta,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn
            }
            results.append(result)
            
            # Check if this is the best so far
            if f1 > best_f1:
                best_f1 = f1
                best_f1_thresholds = (window_threshold, element_threshold)
            
            if precision > best_precision and recall >= 0.4:  # Require at least 40% recall
                best_precision = precision
                best_precision_thresholds = (window_threshold, element_threshold)
            
            if f_beta > best_fbeta:
                best_fbeta = f_beta
                best_fbeta_thresholds = (window_threshold, element_threshold)
            
            # Print progress
            print(f"W:{w_factor:.1f} E:{e_factor:.1f} - P:{precision:.4f} R:{recall:.4f} F1:{f1:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Extract values for plotting
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    # Create scatter plot
    sc = plt.scatter(recalls, precisions, c=f1s, cmap='viridis', s=100, alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('F1 Score')
    
    # Mark best points
    if best_f1_thresholds:
        best_f1_result = next((r for r in results 
                             if abs(r['window_threshold'] - best_f1_thresholds[0]) < 1e-6 
                             and abs(r['element_threshold'] - best_f1_thresholds[1]) < 1e-6), None)
        if best_f1_result:
            plt.scatter([best_f1_result['recall']], [best_f1_result['precision']], 
                       s=200, facecolors='none', edgecolors='red', 
                       linewidth=2, label=f'Best F1: {best_f1:.4f}')
    
    if best_precision_thresholds:
        best_prec_result = next((r for r in results 
                               if abs(r['window_threshold'] - best_precision_thresholds[0]) < 1e-6 
                               and abs(r['element_threshold'] - best_precision_thresholds[1]) < 1e-6), None)
        if best_prec_result:
            plt.scatter([best_prec_result['recall']], [best_prec_result['precision']], 
                       s=200, facecolors='none', edgecolors='blue', 
                       linewidth=2, label=f'Best Precision@40%: {best_precision:.4f}')
    
    if best_fbeta_thresholds:
        best_fbeta_result = next((r for r in results 
                                if abs(r['window_threshold'] - best_fbeta_thresholds[0]) < 1e-6 
                                and abs(r['element_threshold'] - best_fbeta_thresholds[1]) < 1e-6), None)
        if best_fbeta_result:
            plt.scatter([best_fbeta_result['recall']], [best_fbeta_result['precision']], 
                       s=200, facecolors='none', edgecolors='green', 
                       linewidth=2, label=f'Best F-beta: {best_fbeta:.4f}')
    
    # Add annotations for some points
    for i, result in enumerate(results):
        if i % 8 == 0:  # Annotate every 8th point to avoid clutter
            plt.annotate(f"W:{result['window_factor']:.1f} E:{result['element_factor']:.1f}",
                        (result['recall'], result['precision']),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Different Threshold Combinations')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'threshold_combinations.png'))
    plt.close()
    
    # Save results
    with open(os.path.join(output_dir, 'threshold_evaluation.json'), 'w') as f:
        json.dump({
            'results': results,
            'best_f1': {
                'f1': best_f1,
                'window_threshold': best_f1_thresholds[0] if best_f1_thresholds else None,
                'element_threshold': best_f1_thresholds[1] if best_f1_thresholds else None
            },
            'best_precision': {
                'precision': best_precision,
                'window_threshold': best_precision_thresholds[0] if best_precision_thresholds else None,
                'element_threshold': best_precision_thresholds[1] if best_precision_thresholds else None
            },
            'best_fbeta': {
                'f_beta': best_fbeta,
                'window_threshold': best_fbeta_thresholds[0] if best_fbeta_thresholds else None,
                'element_threshold': best_fbeta_thresholds[1] if best_fbeta_thresholds else None
            }
        }, f, indent=2)
    
    print("\nBest threshold combinations:")
    print(f"Best F1 Score ({best_f1:.4f}): Window={best_f1_thresholds[0]:.4f}, Element={best_f1_thresholds[1]:.4f}")
    print(f"Best Precision@40% ({best_precision:.4f}): Window={best_precision_thresholds[0]:.4f}, Element={best_precision_thresholds[1]:.4f}")
    print(f"Best F-beta Score ({best_fbeta:.4f}): Window={best_fbeta_thresholds[0]:.4f}, Element={best_fbeta_thresholds[1]:.4f}")
    
    print(f"\nResults saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Threshold Combinations')
    parser.add_argument('--window_model', type=str, required=True, help='Window-level model file')
    parser.add_argument('--element_model', type=str, required=True, help='Element-level model file')
    parser.add_argument('--data_file', type=str, required=True, help='HDF5 data file')
    parser.add_argument('--output_dir', type=str, default='threshold_evaluation', help='Output directory')
    
    args = parser.parse_args()
    
    evaluate_thresholds(args.window_model, args.element_model, args.data_file, args.output_dir)

if __name__ == "__main__":
    main()