import json
import matplotlib.pyplot as plt
import numpy as np
import h5py
from preprocessing import normalize_image
import cv2
import os
import argparse

def analyze_results(results_file, dataset_file, output_dir="analysis_results"):
    """Analyze detection results and create visualizations"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract metrics
    overall_metrics = results['overall_metrics']
    image_results = results['image_results']
    
    # Load dataset for ground truth comparison
    with h5py.File(dataset_file, 'r') as f:
        images = f['images'][:]
        dead_elements = f['dead_elements'][:]
    
    # 1. Analyze detection distribution
    detection_counts = []
    truth_counts = []
    precision_by_image = []
    recall_by_image = []
    
    for i, result in enumerate(image_results):
        # Count detections
        detection_counts.append(result['num_defects'])
        
        # Count ground truth defects
        if i < len(dead_elements):
            truth_count = np.sum(dead_elements[i] > 0)
            truth_counts.append(truth_count)
        
        # Calculate per-image metrics if available
        if result['metrics'] and 'precision' in result['metrics'] and truth_count > 0:
            precision = result['metrics']['precision']
            recall = result['metrics']['recall']
            
            if precision > 0:
                precision_by_image.append(precision)
            if recall > 0:
                recall_by_image.append(recall)
    
    # 2. Plot detection distribution
    plt.figure(figsize=(12, 6))
    plt.hist([truth_counts, detection_counts], bins=range(max(max(truth_counts, default=0), max(detection_counts, default=0))+2),
             alpha=0.7, label=['Ground Truth', 'Detected'])
    plt.xlabel('Number of Defects per Image')
    plt.ylabel('Number of Images')
    plt.title('Defect Distribution Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'defect_distribution.png'))
    plt.close()
    
    # 3. Analyze confidence scores
    all_confidences = []
    tp_confidences = []
    fp_confidences = []
    
    for result in image_results:
        if result['defect_positions']:
            for pos, conf in result['defect_positions']:
                all_confidences.append(conf)
                
                # Check if true positive or false positive
                if 'metrics' in result and result['metrics']:
                    image_idx = result['image_index']
                    if image_idx < len(dead_elements):
                        if pos < len(dead_elements[image_idx]) and dead_elements[image_idx][pos] > 0:
                            tp_confidences.append(conf)
                        else:
                            fp_confidences.append(conf)
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    if tp_confidences and fp_confidences:
        plt.hist([tp_confidences, fp_confidences], bins=20,
                 alpha=0.7, label=['True Positives', 'False Positives'])
    else:
        plt.hist(all_confidences, bins=20, alpha=0.7, label=['All Detections'])
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Detection Confidence Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()
    
    # 4. Analyze position distribution
    tp_positions = []
    fp_positions = []
    fn_positions = []
    
    for i, result in enumerate(image_results):
        # Extract detected positions
        detected_positions = [pos for pos, _ in result['defect_positions']]
        
        # Get ground truth for this image
        if i < len(dead_elements):
            true_positions = np.where(dead_elements[i] > 0)[0]
            
            # Identify TPs, FPs, and FNs
            for pos in detected_positions:
                if pos < len(dead_elements[i]) and dead_elements[i][pos] > 0:
                    tp_positions.append(pos)
                else:
                    fp_positions.append(pos)
            
            for pos in true_positions:
                if pos not in detected_positions:
                    fn_positions.append(pos)
    
    # Plot position distribution
    plt.figure(figsize=(14, 6))
    bins = np.arange(0, 130, 5)  # Assumes transducer has ~128 elements
    
    if tp_positions:
        plt.hist(tp_positions, bins=bins, alpha=0.6, label='True Positives', color='green')
    if fp_positions:
        plt.hist(fp_positions, bins=bins, alpha=0.6, label='False Positives', color='red')
    if fn_positions:
        plt.hist(fn_positions, bins=bins, alpha=0.6, label='False Negatives', color='blue')
    
    plt.xlabel('Element Position')
    plt.ylabel('Count')
    plt.title('Detection Distribution by Position')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'position_distribution.png'))
    plt.close()
    
    # 5. Find representative examples
    print("Finding representative examples...")
    
    # Find images with both TPs and FNs
    mixed_result_images = []
    for i, result in enumerate(image_results):
        if 'metrics' in result and result['metrics']:
            tp = result['metrics'].get('true_positives', 0)
            fn = result['metrics'].get('false_negatives', 0)
            if tp > 0 and fn > 0:
                mixed_result_images.append(i)
    
    # Save a few representative mixed images
    if mixed_result_images:
        for idx in mixed_result_images[:5]:  # Take up to 5 examples
            img = images[idx]
            if img.max() > 1.0:
                img = normalize_image(img)
            else:
                img = (img * 255).astype(np.uint8)
            
            # Draw detections
            result = image_results[idx]
            detected = [pos for pos, _ in result['defect_positions']]
            true_defects = np.where(dead_elements[idx] > 0)[0]
            
            # Create visualization
            vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            height = vis_img.shape[0]
            
            # Draw false negatives (missed defects) - blue
            for pos in true_defects:
                if pos not in detected:
                    cv2.line(vis_img, (pos, 0), (pos, height//3), (255, 0, 0), 2)
            
            # Draw true positives - green
            for pos in detected:
                if pos < len(dead_elements[idx]) and dead_elements[idx][pos] > 0:
                    cv2.line(vis_img, (pos, 0), (pos, height//3), (0, 255, 0), 2)
            
            # Draw false positives - red
            for pos in detected:
                if pos >= len(dead_elements[idx]) or dead_elements[idx][pos] == 0:
                    cv2.line(vis_img, (pos, 0), (pos, height//3), (0, 0, 255), 2)
            
            # Add legend
            cv2.putText(vis_img, "Green: TP", (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(vis_img, "Blue: FN", (10, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(vis_img, "Red: FP", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imwrite(os.path.join(output_dir, f"mixed_example_{idx}.png"), vis_img)
    
    # 6. Recall analysis - find common patterns in missed defects
    print("Analyzing missed defects...")
    
    # Take sample of false negatives and extract their patterns
    if fn_positions:
        fn_sample_indices = np.random.choice(len(fn_positions), min(100, len(fn_positions)), replace=False)
        fn_samples = [fn_positions[i] for i in fn_sample_indices]
        
        # Extract image patterns for these positions
        fn_patterns = []
        for i, result in enumerate(image_results):
            if 'metrics' in result and result['metrics'] and result['metrics'].get('false_negatives', 0) > 0:
                img_idx = result['image_index']
                if img_idx < len(images):
                    img = images[img_idx]
                    true_defects = np.where(dead_elements[img_idx] > 0)[0]
                    detected = [pos for pos, _ in result['defect_positions']]
                    
                    # Find missed defects in this image
                    for pos in true_defects:
                        if pos not in detected and pos < img.shape[1]:
                            # Extract vertical profile
                            profile = img[:, pos]
                            fn_patterns.append(profile)
                            
                            if len(fn_patterns) >= 100:
                                break
                    
                    if len(fn_patterns) >= 100:
                        break
        
        # Plot average pattern of missed defects
        if fn_patterns:
            max_len = max(len(p) for p in fn_patterns) if fn_patterns else 0
            aligned_patterns = []
            for p in fn_patterns:
                if len(p) < max_len:
                    # Pad pattern to match longest
                    p_padded = np.pad(p, (0, max_len - len(p)), 'constant', constant_values=0)
                    aligned_patterns.append(p_padded)
                else:
                    aligned_patterns.append(p)
            
            avg_pattern = np.mean(aligned_patterns, axis=0)
            std_pattern = np.std(aligned_patterns, axis=0)
            
            plt.figure(figsize=(8, 6))
            plt.plot(avg_pattern, label='Average Pattern', color='blue')
            plt.fill_between(
                range(len(avg_pattern)),
                avg_pattern - std_pattern,
                avg_pattern + std_pattern,
                alpha=0.3,
                color='blue'
            )
            plt.title('Average Pattern of Missed Defects')
            plt.xlabel('Depth')
            plt.ylabel('Intensity')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'missed_defect_pattern.png'))
            plt.close()
    
    # 7. Summary report
    summary = {
        'overall_metrics': overall_metrics,
        'detection_distribution': {
            'min_detections': min(detection_counts) if detection_counts else 0,
            'max_detections': max(detection_counts) if detection_counts else 0,
            'avg_detections': float(np.mean(detection_counts)) if detection_counts else 0,
            'total_detections': sum(detection_counts) if detection_counts else 0
        },
        'truth_distribution': {
            'min_defects': min(truth_counts) if truth_counts else 0,
            'max_defects': max(truth_counts) if truth_counts else 0,
            'avg_defects': float(np.mean(truth_counts)) if truth_counts else 0,
            'total_defects': sum(truth_counts) if truth_counts else 0
        },
        'confidence_stats': {
            'min_conf': float(min(all_confidences)) if all_confidences else 0,
            'max_conf': float(max(all_confidences)) if all_confidences else 0,
            'avg_conf': float(np.mean(all_confidences)) if all_confidences else 0
        },
        'position_analysis': {
            'true_positives': len(tp_positions),
            'false_positives': len(fp_positions),
            'false_negatives': len(fn_positions)
        }
    }
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis complete. Results saved to {output_dir}.")
    return summary

def main():
    parser = argparse.ArgumentParser(description='Analyze Detection Results')
    parser.add_argument('--results', type=str, required=True, help='Detection results JSON file')
    parser.add_argument('--dataset', type=str, required=True, help='Original dataset HDF5 file')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='Output directory')
    
    args = parser.parse_args()
    analyze_results(args.results, args.dataset, args.output_dir)

if __name__ == "__main__":
    main()