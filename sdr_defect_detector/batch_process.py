import argparse
import numpy as np
import h5py
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd
import seaborn as sns

from preprocessing import normalize_image, extract_roi, create_windows
from feature_extraction import extract_window_features
from visualization import visualize_defects
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

def analyze_confidence_distribution(results):
    """Analyze confidence distributions of true/false positives"""
    true_positives_conf = []
    false_positives_conf = []
    
    for result in results:
        if result['metrics'] is not None:
            pred_pos = [pos for pos, _ in result['defect_positions']]
            pred_conf = [conf for _, conf in result['defect_positions']]
            true_pos = result['true_defects']
            
            for i, (pos, conf) in enumerate(zip(pred_pos, pred_conf)):
                if pos in true_pos:
                    true_positives_conf.append(conf)
                else:
                    false_positives_conf.append(conf)
    
    # Create distribution plots
    plt.figure(figsize=(12, 6))
    
    if true_positives_conf and false_positives_conf:
        # Create dataframe for seaborn
        tp_df = pd.DataFrame({'confidence': true_positives_conf, 'type': 'True Positive'})
        fp_df = pd.DataFrame({'confidence': false_positives_conf, 'type': 'False Positive'})
        df = pd.concat([tp_df, fp_df])
        
        # Plot distributions
        sns.histplot(data=df, x='confidence', hue='type', bins=20, 
                    alpha=0.6, element="step", common_norm=False)
        plt.title('Confidence Distribution for True vs False Positives')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
    else:
        if true_positives_conf:
            plt.hist(true_positives_conf, bins=20, alpha=0.6, label='True Positives')
        if false_positives_conf:
            plt.hist(false_positives_conf, bins=20, alpha=0.6, label='False Positives')
        
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(alpha=0.3)
    
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description='Batch Process Ultrasound Transducer Images with Enhanced Precision')
    parser.add_argument('--model', type=str, default='defect_detection_model.pkl', help='Model file path')
    parser.add_argument('--data_file', type=str, required=True, help='HDF5 data file containing images')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for feature extraction')
    parser.add_argument('--overlap', type=float, default=0.5, help='Window overlap ratio')
    parser.add_argument('--threshold', type=float, help='Custom threshold (overrides saved threshold)')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index in dataset')
    parser.add_argument('--end_idx', type=int, default=-1, help='End index in dataset (-1 for all)')
    parser.add_argument('--save_vis', action='store_true', help='Save visualizations')
    parser.add_argument('--precision_mode', action='store_true', 
                      help='Enable precision mode (stricter detection criteria)')
    parser.add_argument('--adjustment_factor', type=float, default=1.0,
                      help='Adjust threshold by this factor (> 1.0 increases precision)')
    parser.add_argument('--sensitivity', choices=['high', 'medium', 'low'], default='medium',
                      help='Detection sensitivity: high (more defects), medium (balanced), low (fewer false positives)')
    
    args = parser.parse_args()
    
    # Set threshold adjustment based on sensitivity
    if args.sensitivity == 'high':
        threshold_factor = max(0.8, args.adjustment_factor)
        precision_mode = False
    elif args.sensitivity == 'low':
        threshold_factor = max(1.2, args.adjustment_factor)
        precision_mode = True
    else:  # medium
        threshold_factor = args.adjustment_factor
        precision_mode = args.precision_mode
    
    # Load model
    model, threshold, model_data = load_model(args.model)
    
    # Override threshold if specified
    if args.threshold is not None:
        threshold = args.threshold
    
    # Apply sensitivity-based adjustment
    threshold = threshold * threshold_factor
    print(f"Using threshold: {threshold:.4f} (adjustment factor: {threshold_factor:.2f})")
    print(f"Precision mode: {'Enabled' if precision_mode else 'Disabled'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images from HDF5 file
    print(f"Loading data from {args.data_file}")
    with h5py.File(args.data_file, 'r') as f:
        if 'images' not in f:
            print(f"Error: No 'images' dataset found in {args.data_file}")
            return
        
        # Determine range of images to process
        num_images = len(f['images'])
        start_idx = args.start_idx
        end_idx = num_images if args.end_idx == -1 else min(args.end_idx, num_images)
        
        print(f"Processing images from index {start_idx} to {end_idx-1} (total: {end_idx-start_idx})")
        
        # Check if ground truth is available
        has_ground_truth = 'dead_elements' in f
        
        # Prepare results storage
        results = []
        overall_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        # Process each image
        for idx in tqdm(range(start_idx, end_idx)):
            image = f['images'][idx]
            
            # Get ground truth if available
            true_defects = []
            if has_ground_truth:
                dead_elements = f['dead_elements'][idx]
                true_defects = np.where(dead_elements > 0)[0].tolist()
            
            # Get filename if available
            filename = f"image_{idx}"
            if 'filenames' in f:
                try:
                    filename = f['filenames'][idx].decode('utf-8')
                except:
                    filename = str(f['filenames'][idx])
            
            # Detect defects with improved precision
            defect_positions = detect_defects_in_image(
                image, model, threshold, 
                window_size=args.window_size, 
                overlap=args.overlap,
                precision_mode=precision_mode
            )
            
            # Calculate metrics if ground truth available
            metrics = None
            if has_ground_truth:
                metrics = calculate_metrics(defect_positions, true_defects)
                
                # Update overall metrics
                overall_metrics['true_positives'] += metrics['true_positives']
                overall_metrics['false_positives'] += metrics['false_positives']
                overall_metrics['false_negatives'] += metrics['false_negatives']
            
            # Create image result entry
            result = {
                'image_index': idx,
                'filename': filename,
                'num_defects_detected': len(defect_positions),
                'defect_positions': [(int(pos), float(conf)) for pos, conf in defect_positions],
                'true_defects': true_defects if has_ground_truth else None,
                'metrics': metrics
            }
            results.append(result)
            
            # Save visualization if requested
            if args.save_vis:
                # Convert image to 8-bit grayscale if needed
                if image.max() > 1.0:
                    vis_image_base = normalize_image(image)
                else:
                    vis_image_base = (image * 255).astype(np.uint8)
                
                vis_image = visualize_defects(vis_image_base, defect_positions)
                
                # Overlay ground truth if available
                if true_defects:
                    height = vis_image.shape[0]
                    for pos in true_defects:
                        # Draw green ground truth markers
                        cv2.line(vis_image, (pos, 0), (pos, height//8), (0, 255, 0), 1)
                        cv2.circle(vis_image, (pos, height//6), 3, (0, 255, 0), 1)
                
                vis_filename = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_vis.png")
                cv2.imwrite(vis_filename, vis_image)
        
        # Calculate overall metrics
        if has_ground_truth:
            total_tp = overall_metrics['true_positives']
            total_fp = overall_metrics['false_positives']
            total_fn = overall_metrics['false_negatives']
            
            overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
            
            overall_metrics['precision'] = overall_precision
            overall_metrics['recall'] = overall_recall
            overall_metrics['f1'] = overall_f1
            
            print("\nOverall Performance:")
            print(f"Precision: {overall_precision:.4f}")
            print(f"Recall: {overall_recall:.4f}")
            print(f"F1 Score: {overall_f1:.4f}")
            print(f"True Positives: {total_tp}, False Positives: {total_fp}, False Negatives: {total_fn}")
        
        # Save results to JSON
        results_summary = {
            'model_filename': args.model,
            'threshold': float(threshold),
            'window_size': args.window_size,
            'overlap': args.overlap,
            'precision_mode': precision_mode,
            'images_processed': end_idx - start_idx,
            'overall_metrics': overall_metrics if has_ground_truth else None,
            'image_results': results
        }
        
        results_filename = os.path.join(args.output_dir, 'detection_results.json')
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        
        with open(results_filename, 'w') as f:
            json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nResults saved to {results_filename}")
        
        # Generate enhanced summary plots if ground truth is available
        if has_ground_truth and len(results) > 0:
            # 1. Distribution of defects per image
            plt.figure(figsize=(12, 7))
            
            detected_counts = [len(r['defect_positions']) for r in results]
            true_counts = [len(r['true_defects']) for r in results]
            
            plt.hist([true_counts, detected_counts], bins=range(max(max(true_counts), max(detected_counts)) + 2), 
                     alpha=0.7, label=['Ground Truth', 'Detected'])
            plt.xlabel('Number of Defects')
            plt.ylabel('Number of Images')
            plt.title('Distribution of Defects per Image')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(args.output_dir, 'defect_distribution.png'))
            plt.close()
            
            # 2. Precision-Recall by image
            all_precisions = [r['metrics']['precision'] for r in results if r['metrics'] is not None and r['metrics']['precision'] > 0]
            all_recalls = [r['metrics']['recall'] for r in results if r['metrics'] is not None and r['metrics']['recall'] > 0]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(all_recalls, all_precisions, alpha=0.7)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision vs Recall by Image')
            plt.grid(alpha=0.3)
            plt.xlim(0, 1.05)
            plt.ylim(0, 1.05)
            plt.savefig(os.path.join(args.output_dir, 'precision_recall.png'))
            plt.close()
            
            # 3. F1 Distribution
            all_f1s = [r['metrics']['f1'] for r in results if r['metrics'] is not None]
            
            plt.figure(figsize=(10, 6))
            plt.hist(all_f1s, bins=10, alpha=0.7)
            plt.xlabel('F1 Score')
            plt.ylabel('Number of Images')
            plt.title('Distribution of F1 Scores')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(args.output_dir, 'f1_distribution.png'))
            plt.close()
            
            # 4. Confidence vs Accuracy plot
            all_confs = []
            all_correct = []
            
            for r in results:
                if r['metrics'] is not None:
                    pred_pos = [pos for pos, _ in r['defect_positions']]
                    true_pos = r['true_defects']
                    
                    for pos, conf in r['defect_positions']:
                        all_confs.append(conf)
                        all_correct.append(1 if pos in true_pos else 0)
            
            if all_confs:
                # Sort by confidence
                conf_acc = sorted(zip(all_confs, all_correct), key=lambda x: x[0])
                confs = [c for c, _ in conf_acc]
                correct = [c for _, c in conf_acc]
                
                # Calculate moving average accuracy
                window_size = min(50, len(correct))
                moving_acc = []
                
                for i in range(len(correct) - window_size + 1):
                    moving_acc.append(sum(correct[i:i+window_size]) / window_size)
                
                plt.figure(figsize=(12, 6))
                plt.scatter(confs[window_size-1:], moving_acc, alpha=0.5, s=10)
                plt.xlabel('Confidence Score')
                plt.ylabel('Accuracy (Moving Average)')
                plt.title(f'Detection Accuracy vs Confidence ({window_size}-sample moving average)')
                plt.grid(alpha=0.3)
                plt.xlim(0, 1.05)
                plt.ylim(0, 1.05)
                plt.savefig(os.path.join(args.output_dir, 'confidence_accuracy.png'))
                plt.close()
            
            # 5. New analysis: Confidence distribution for true/false positives
            conf_dist_fig = analyze_confidence_distribution(results)
            conf_dist_fig.savefig(os.path.join(args.output_dir, 'confidence_distribution.png'))
            plt.close()
            
            # 6. Position analysis: Where are defects commonly occurring?
            all_true_defect_positions = []
            all_false_positive_positions = []
            
            for r in results:
                if r['metrics'] is not None:
                    all_true_defect_positions.extend(r['true_defects'])
                    pred_pos = [pos for pos, _ in r['defect_positions']]
                    for pos in pred_pos:
                        if pos not in r['true_defects']:
                            all_false_positive_positions.append(pos)
            
            if all_true_defect_positions or all_false_positive_positions:
                plt.figure(figsize=(12, 6))
                
                # Plot positional distribution
                if all_true_defect_positions:
                    plt.hist(all_true_defect_positions, bins=range(0, 129, 4), alpha=0.7, 
                            label='True Defects', color='green')
                
                if all_false_positive_positions:
                    plt.hist(all_false_positive_positions, bins=range(0, 129, 4), alpha=0.5, 
                            label='False Positives', color='red')
                
                plt.xlabel('Transducer Position')
                plt.ylabel('Count')
                plt.title('Distribution of Defects by Position')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(args.output_dir, 'position_distribution.png'))
                plt.close()
            
            print(f"Summary plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()