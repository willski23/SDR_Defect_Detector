import argparse
import numpy as np
import h5py
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from preprocessing import normalize_image
from element_features import simple_element_features
from visualization import visualize_defects
from detect_image import extract_transducer_profile

def load_models(window_model_path, element_model_path=None):
    """Load both window-level and element-level models"""
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

def detect_windows(image, model, threshold, window_size=8, overlap=0.5):
    """First stage: Detect suspicious windows with standard model"""
    from detect_image import detect_defects_in_image
    
    # Use existing detection function with precision mode
    detected_positions = detect_defects_in_image(
        image, model, threshold, 
        window_size=window_size, 
        overlap=overlap,
        precision_mode=True
    )
    
    return detected_positions

def detect_edge_defects(image, element_model, element_threshold):
    """Special detection path for edge elements"""
    # Ensure image is normalized
    if image.max() > 1.0:
        norm_image = normalize_image(image)
    else:
        norm_image = (image * 255).astype(np.uint8)
    
    edge_positions = []
    edge_width = 10  # Consider first/last 10 elements as edges
    
    # Check left edge
    for pos in range(edge_width):
        if pos < norm_image.shape[1]:
            features = simple_element_features(norm_image, pos)
            element_proba = element_model.predict_proba([features])[0, 1]
            
            # Use lower threshold for edge elements
            if element_proba >= element_threshold * 0.7:
                edge_positions.append((pos, element_proba))
    
    # Check right edge
    for pos in range(norm_image.shape[1] - edge_width, norm_image.shape[1]):
        if pos >= 0:
            features = simple_element_features(norm_image, pos)
            element_proba = element_model.predict_proba([features])[0, 1]
            
            # Use lower threshold for edge elements
            if element_proba >= element_threshold * 0.7:
                edge_positions.append((pos, element_proba))
    
    return edge_positions

def refine_element_detection(image, suspicious_positions, element_model, element_threshold):
    """Second stage: Refine detection at element level"""
    if not element_model or not suspicious_positions:
        return suspicious_positions
    
    # Ensure image is normalized
    if image.max() > 1.0:
        norm_image = normalize_image(image)
    else:
        norm_image = (image * 255).astype(np.uint8)
    
    # Extract element-level features for each suspicious position
    refined_positions = []
    
    for pos, window_conf in suspicious_positions:
        try:
            # Extract features for this specific element
            element_features = simple_element_features(norm_image, pos)
            
            # Predict with element classifier
            element_proba = element_model.predict_proba([element_features])[0, 1]
            
            # Only keep if element classifier also confirms with high confidence
            if element_proba >= element_threshold:
                # Combine confidences with emphasis on element-level confidence
                combined_conf = 0.3 * window_conf + 0.7 * element_proba
                refined_positions.append((pos, combined_conf))
        except Exception as e:
            # If error occurs, fall back to original confidence but only for very confident detections
            if window_conf > element_threshold * 1.5:  # Higher threshold for fallback
                refined_positions.append((pos, window_conf * 0.8))  # Reduce confidence slightly
    
    return refined_positions

def apply_post_processing(image, defect_positions):
    """Apply post-processing filters for maximum precision"""
    if not defect_positions:
        return []
        
    filtered_positions = []
    
    # 1. Extract intensity profile for validation
    profile = extract_transducer_profile(image)
    
    # 2. Edge filtering (stricter near edges)
    image_width = image.shape[1]
    edge_margin = int(image_width * 0.1)  # 10% from each edge
    
    # 3. Process each candidate position
    for pos, conf in defect_positions:
        # Skip if position is out of bounds
        if pos >= len(profile) or pos < 0:
            continue
            
        # Check if position is at edge (common false positive location)
        is_at_edge = pos < edge_margin or pos >= image_width - edge_margin
        edge_threshold_factor = 0.8 if is_at_edge else 1.0        
        # Check if position is a local minimum in intensity (defects usually are)
        is_local_minimum = False
        window_size = 5
        start = max(0, pos - window_size//2)
        end = min(len(profile), pos + window_size//2 + 1)
        
        if end > start and end <= len(profile):
            local_profile = profile[start:end]
            if len(local_profile) > 0:
                min_pos = start + np.argmin(local_profile)
                min_value = local_profile[min_pos - start] if min_pos - start < len(local_profile) else 1.0
                
                # Check if position is a significant minimum (not just noise)
                is_significant_minimum = min_value < 0.5  # Below 50% intensity
                is_local_minimum = abs(min_pos - pos) <= 1 and is_significant_minimum
        
        # Check for isolation (defects tend to be isolated)
        isolation_distances = []
        for other_pos, _ in defect_positions:
            if other_pos != pos:
                isolation_distances.append(abs(other_pos - pos))
        
        is_isolated = not isolation_distances or min(isolation_distances + [999]) > 3
        
        # Analyze vertical pattern
        min_vertical_intensity = 0.0
        if pos < image.shape[1]:
            element_col = image[:, pos]
            if len(element_col) > 0:
                min_vertical_intensity = np.min(element_col) / (np.max(element_col) + 1e-8)
        
        # Final decision with multiple criteria
        keep_detection = (
            # Must meet at least one physical criterion
            (is_local_minimum or min_vertical_intensity < 0.3) and
            # Must meet confidence criterion
            (conf > 0.7 * edge_threshold_factor) and 
            # Must not be suspicious (multiple detections in close proximity)
            (is_isolated or conf > 0.9)
        )
        
        if keep_detection:
            filtered_positions.append((pos, conf))
    
    return filtered_positions

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

def high_precision_detection(image, window_model, window_threshold, 
                           element_model=None, element_threshold=None,
                           window_size=8, overlap=0.5):
    """Perform two-stage high precision detection"""
    # Stage 1: Detect suspicious windows/elements
    suspicious_positions = detect_windows(
        image, window_model, window_threshold, window_size, overlap
    )
    
    # Stage 2: Refine with element-level classifier (if available)
    if element_model:
        refined_positions = refine_element_detection(
            image, suspicious_positions, element_model, element_threshold
        )
    else:
        refined_positions = suspicious_positions
    
    # Stage 3: Apply post-processing filters
    defect_positions = apply_post_processing(image, defect_positions)
    
    # Special edge detection path
    if element_model:
        edge_detections = detect_edge_defects(image, element_model, element_threshold)
        
        # Combine with main detections (avoid duplicates)
        detected_positions = set(pos for pos, _ in defect_positions)
        for pos, conf in edge_detections:
            if pos not in detected_positions:
                defect_positions.append((pos, conf))
    
    return defect_positions

def main():
    parser = argparse.ArgumentParser(description='High-Precision Defect Detection')
    parser.add_argument('--window_model', type=str, required=True, help='Window-level model file')
    parser.add_argument('--element_model', type=str, help='Element-level model file (optional)')
    parser.add_argument('--data_file', type=str, required=True, help='HDF5 data file')
    parser.add_argument('--output_dir', type=str, default='high_precision_output', help='Output directory')
    parser.add_argument('--window_size', type=int, default=8, help='Window size')
    parser.add_argument('--overlap', type=float, default=0.5, help='Window overlap')
    parser.add_argument('--save_vis', action='store_true', help='Save visualizations')
    parser.add_argument('--window_threshold_factor', type=float, default=1.0, 
                       help='Factor to adjust window threshold')
    parser.add_argument('--element_threshold_factor', type=float, default=1.0,
                       help='Factor to adjust element threshold')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    window_model, window_threshold, element_model, element_threshold = load_models(
        args.window_model, args.element_model
    )
    
    # Apply threshold adjustments
    window_threshold *= args.window_threshold_factor
    if element_threshold:
        element_threshold *= args.element_threshold_factor
    
    print(f"Using window threshold: {window_threshold:.4f}")
    if element_threshold:
        print(f"Using element threshold: {element_threshold:.4f}")
    
    # Load data
    print(f"Loading data from {args.data_file}")
    with h5py.File(args.data_file, 'r') as f:
        images = f['images'][:]
        
        # Load true defect positions if available
        has_ground_truth = 'dead_elements' in f
        true_defects = []
        
        if has_ground_truth:
            for elements in f['dead_elements'][:]:
                positions = np.where(elements > 0)[0].tolist()
                true_defects.append(positions)
        
        # Process images
        results = []
        
        # Prepare metrics tracking
        metrics_sum = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        print(f"Processing {len(images)} images...")
        for i, image in enumerate(tqdm(images, desc="Processing images")):
            # High precision detection
            defect_positions = high_precision_detection(
                image, window_model, window_threshold,
                element_model, element_threshold,
                args.window_size, args.overlap
            )
            
            # Calculate metrics if ground truth available
            if has_ground_truth and i < len(true_defects):
                image_metrics = calculate_metrics(defect_positions, true_defects[i])
                
                # Update overall metrics
                metrics_sum['true_positives'] += image_metrics['true_positives']
                metrics_sum['false_positives'] += image_metrics['false_positives']
                metrics_sum['false_negatives'] += image_metrics['false_negatives']
            else:
                image_metrics = None
            
            # Store results
            results.append({
                'image_index': i,
                'num_defects': len(defect_positions),
                'defect_positions': [(int(pos), float(conf)) for pos, conf in defect_positions],
                'metrics': image_metrics
            })
            
            # Save visualization if requested
            if args.save_vis:
                if image.max() > 1.0:
                    vis_image = normalize_image(image)
                else:
                    vis_image = (image * 255).astype(np.uint8)
                
                vis_output = visualize_defects(vis_image, defect_positions)
                
                # Add ground truth if available
                if has_ground_truth and i < len(true_defects):
                    height = vis_output.shape[0]
                    for pos in true_defects[i]:
                        cv2.line(vis_output, (pos, 0), (pos, height//8), (0, 255, 0), 1)
                        cv2.circle(vis_output, (pos, height//6), 3, (0, 255, 0), 1)
                
                cv2.imwrite(os.path.join(args.output_dir, f"image_{i}_vis.png"), vis_output)
        
        # Calculate overall metrics
        if has_ground_truth:
            total_tp = metrics_sum['true_positives']
            total_fp = metrics_sum['false_positives']
            total_fn = metrics_sum['false_negatives']
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f_beta = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall) if (precision + recall) > 0 else 0
            
            print("\nOverall Performance:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"F-beta (β=0.5): {f_beta:.4f}")
            print(f"True Positives: {total_tp}")
            print(f"False Positives: {total_fp}")
            print(f"False Negatives: {total_fn}")
            
            # Save metrics
            overall_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'f_beta': f_beta,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn
            }
        else:
            overall_metrics = None
        
        # Save results
        with open(os.path.join(args.output_dir, 'detection_results.json'), 'w') as f:
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
            
            json.dump({
                'overall_metrics': overall_metrics,
                'image_results': results
            }, f, indent=2, cls=NumpyEncoder)
        
        print(f"Results saved to {args.output_dir}/detection_results.json")

if __name__ == "__main__":
    main()