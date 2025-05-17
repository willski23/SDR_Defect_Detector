import argparse
import numpy as np
import h5py
import cv2
import pickle
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from preprocessing import normalize_image, extract_roi, create_windows
from feature_extraction import extract_window_features, safe_correlation
from visualization import visualize_defects

def load_model(filename='defect_detection_model.pkl'):
    """Load saved model"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    threshold = model_data['threshold']
    
    print(f"Loaded model trained on {model_data['timestamp']}")
    if 'metrics' in model_data and 'element_level' in model_data['metrics']:
        print(f"Model performance: F1={model_data['metrics']['element_level'].get('f1', 'N/A'):.4f}")
    
    return model, threshold, model_data

def extract_transducer_profile(image):
    """Extract transducer intensity profile for advanced detection logic with improved robustness"""
    # Get top portion for analysis
    height, width = image.shape
    if height == 0 or width == 0:
        return np.array([])
        
    top_section_height = min(int(height * 0.15), 30)
    if top_section_height <= 0:
        return np.array([])
        
    top_section = image[:top_section_height, :]
    
    # Create intensity profile
    profile = np.mean(top_section, axis=0)
    
    # Normalize profile with safety checks
    profile_min = np.min(profile)
    profile_max = np.max(profile)
    
    if profile_max > profile_min:
        profile = (profile - profile_min) / (profile_max - profile_min + 1e-8)
    else:
        # Handle constant profile
        profile = np.zeros_like(profile)
    
    return profile

def optimize_position_detection(positions, confidences, profile, window_size=8):
    """Refine defect positions using signal profile analysis with enhanced robustness"""
    refined_positions = []
    
    if len(profile) == 0:
        # Return original positions if profile is empty
        return [(pos, conf) for pos, conf in zip(positions, confidences)]
    
    for position, confidence in zip(positions, confidences):
        # Extract profile segment centered at position
        half_window = window_size // 2
        start = max(0, position - half_window)
        end = min(len(profile), position + half_window + 1)
        
        if start >= end or start >= len(profile) or end <= 0:
            # Skip invalid ranges, but keep original position
            refined_positions.append((position, confidence))
            continue
            
        segment = profile[start:end]
        
        # Look for local minima in the segment
        if len(segment) > 2:
            try:
                # Find local minimum within window
                local_min_idx = start + np.argmin(segment)
                
                # Only adjust position if minimum is significant
                if segment[local_min_idx - start] < np.mean(segment) * 0.9:
                    # Add refined position with original confidence
                    refined_positions.append((local_min_idx, confidence))
                else:
                    # Keep original position
                    refined_positions.append((position, confidence))
            except:
                # Keep original position on error
                refined_positions.append((position, confidence))
        else:
            # Fallback to original position
            refined_positions.append((position, confidence))
    
    return refined_positions

def analyze_pattern_consistency(image, positions):
    """Analyze pattern consistency to filter out false positives with improved robustness"""
    if not positions:
        return []
        
    height, width = image.shape
    roi_height = min(int(height * 0.3), 100)
    if roi_height <= 0:
        return positions
        
    roi = image[:roi_height, :]
    
    # Create vertical profiles for all columns
    profiles = []
    for col in range(width):
        if col < roi.shape[1]:
            profiles.append(roi[:, col])
    
    # Safety check
    if len(profiles) == 0:
        return positions
    
    # Calculate correlation between columns
    valid_positions = []
    
    for pos, conf in positions:
        if pos >= len(profiles) or pos >= width:
            continue
            
        # Get neighboring columns for comparison
        neighbors = []
        for offset in [-4, -3, -2, -1, 1, 2, 3, 4]:
            neighbor_pos = pos + offset
            if 0 <= neighbor_pos < len(profiles):
                neighbors.append(profiles[neighbor_pos])
        
        # Calculate correlations with safe function
        correlations = []
        target_profile = profiles[pos]
        for neighbor in neighbors:
            if len(neighbor) == len(target_profile) and len(target_profile) > 0:
                corr = safe_correlation(target_profile, neighbor)
                correlations.append(corr)
        
        # Check if this column is inconsistent with neighbors
        if correlations and len(correlations) >= 4:
            mean_corr = np.mean(correlations)
            
            # Lower correlation suggests a defect
            if mean_corr < 0.85:
                # Keep position but adjust confidence based on correlation
                adjusted_conf = conf * (1.0 - mean_corr/2)
                valid_positions.append((pos, adjusted_conf))
            else:
                # Likely false positive - very similar to neighbors
                # Only keep if confidence is very high
                if conf > 0.9:
                    valid_positions.append((pos, conf * 0.8))
        else:
            # Not enough neighbors for comparison
            valid_positions.append((pos, conf))
    
    return valid_positions

def detect_defects_in_image(image, model, threshold, window_size=8, overlap=0.5, precision_mode=False):
    """Detect defects in an image with improved precision"""
    # Ensure image is properly normalized
    if image.max() > 1.0:
        norm_image = normalize_image(image)
    else:
        norm_image = (image * 255).astype(np.uint8)
    
    # Extract ROI
    roi = extract_roi(norm_image)
    
    # Create windows
    windows, positions = create_windows(roi, window_size=window_size, overlap=overlap)
    
    # Extract features with enhanced validation
    features = []
    valid_positions = []
    
    for i, (window, pos) in enumerate(zip(windows, positions)):
        try:
            window_features = extract_window_features(window, norm_image, pos)
            
            # Validate features (check for NaN/inf)
            if not np.any(np.isnan(window_features)) and not np.any(np.isinf(window_features)):
                features.append(window_features)
                valid_positions.append(pos)
            else:
                # Replace with zeros to avoid model errors
                fixed_features = np.zeros_like(window_features)
                features.append(fixed_features)
                valid_positions.append(pos)
        except Exception as e:
            # Skip problematic windows
            continue
    
    # Get predictions
    if len(features) > 0:
        X = np.array(features)
        
        # Get probability predictions
        try:
            probas = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"Prediction error: {e}")
            return []
        
        # Initial confidence mapping
        element_probas = {}
        for i, ((start, end), proba) in enumerate(zip(valid_positions, probas)):
            for pos in range(start, end):
                if pos not in element_probas or proba > element_probas[pos]:
                    element_probas[pos] = proba
        
        # Apply strict position-based threshold adjustments
        image_width = roi.shape[1]
        defect_positions = []
        
        # Use higher threshold for precision mode
        base_threshold = threshold * (1.3 if precision_mode else 1.0)
        
        for pos, conf in element_probas.items():
            # Determine adjusted threshold based on position
            if pos < 5 or pos > image_width - 5:  # Edge position
                adjusted_threshold = base_threshold * 1.6  # Much higher threshold at edges
            elif pos < 10 or pos > image_width - 10:  # Near edge
                adjusted_threshold = base_threshold * 1.4  # Higher threshold near edges
            elif pos < 15 or pos > image_width - 15:  # Somewhat near edge
                adjusted_threshold = base_threshold * 1.2  # Slightly higher threshold
            else:
                adjusted_threshold = base_threshold
                
            if conf >= adjusted_threshold:
                defect_positions.append((pos, conf))
        
        # Early return if no defects found
        if not defect_positions:
            return []
        
        # Sort by position for clustering analysis
        defect_positions.sort(key=lambda x: x[0])
        
        # Cluster analysis to identify suspicious detection patterns
        if precision_mode and len(defect_positions) > 1:
            filtered_positions = []
            
            # Identify suspicious patterns (consecutive detections with identical confidence)
            current_confidence = defect_positions[0][1]  # Initialize with first confidence score
            streak_start = 0
            streak_length = 1
            
            for i, (pos, conf) in enumerate(defect_positions):
                # Skip first element since we already initialized with it
                if i == 0:
                    continue
                    
                # Check if this is continuing a streak
                if abs(conf - current_confidence) < 0.0001 and pos == defect_positions[i-1][0] + 1:
                    streak_length += 1
                else:
                    # Process any streak that just ended
                    if streak_length > 3:
                        # For long identical confidence streaks, keep only the strongest point
                        # Extract positions in the streak
                        streak_positions = [defect_positions[j][0] for j in range(streak_start, streak_start + streak_length)]
                        
                        # Extract intensity profile
                        profile = extract_transducer_profile(norm_image)
                        
                        # Find the position with minimum intensity (likely the true defect)
                        if all(p < len(profile) for p in streak_positions):
                            intensities = [profile[p] for p in streak_positions]
                            min_intensity_idx = streak_start + np.argmin(intensities)
                            if min_intensity_idx < len(defect_positions):
                                # Keep only that position, but reduce confidence slightly
                                min_pos, min_conf = defect_positions[min_intensity_idx]
                                filtered_positions.append((min_pos, min_conf * 0.95))
                    else:
                        # For short streaks, keep all positions
                        for j in range(streak_start, streak_start + streak_length):
                            if j >= 0 and j < len(defect_positions):
                                filtered_positions.append(defect_positions[j])
                    
                    # Start new streak
                    streak_start = i
                    streak_length = 1
                    current_confidence = conf
            
            # Handle the final streak
            if streak_length > 3:
                # Similar processing for final streak
                streak_positions = [defect_positions[j][0] for j in range(streak_start, streak_start + streak_length)]
                profile = extract_transducer_profile(norm_image)
                
                if all(p < len(profile) for p in streak_positions):
                    intensities = [profile[p] for p in streak_positions]
                    min_intensity_idx = streak_start + np.argmin(intensities)
                    if min_intensity_idx < len(defect_positions):
                        min_pos, min_conf = defect_positions[min_intensity_idx]
                        filtered_positions.append((min_pos, min_conf * 0.95))
            else:
                # For short streaks, keep all positions
                for j in range(streak_start, streak_start + streak_length):
                    if j >= 0 and j < len(defect_positions):
                        filtered_positions.append(defect_positions[j])
            
            defect_positions = filtered_positions
        
        # Advanced filtering: Analyze pattern consistency
        defect_positions = analyze_pattern_consistency(norm_image, defect_positions)
        
        # Post-processing to remove isolated detections (likely false positives)
        if defect_positions:
            filtered_positions = []
            
            for i, (pos, conf) in enumerate(defect_positions):
                # Check if this is an isolated detection
                is_isolated = True
                for j, (other_pos, _) in enumerate(defect_positions):
                    if i != j and abs(pos - other_pos) <= 3:  # Within 3 elements
                        is_isolated = False
                        break
                
                # Keep non-isolated detections or only very high confidence ones
                if not is_isolated or conf > base_threshold * 1.5:
                    filtered_positions.append((pos, conf))
            
            # If everything was filtered, keep the highest confidence detection
            if not filtered_positions and defect_positions:
                best_pos = max(defect_positions, key=lambda x: x[1])
                filtered_positions.append(best_pos)
            
            # Final position refinement
            profile = extract_transducer_profile(norm_image)
            refined_positions = optimize_position_detection(
                [p[0] for p in filtered_positions],
                [p[1] for p in filtered_positions],
                profile, 
                window_size=window_size
            )
            
            return refined_positions
    
    return []

def main():
    parser = argparse.ArgumentParser(description='Detect Defects in Ultrasound Transducer Image with Enhanced Precision')
    parser.add_argument('--model', type=str, default='defect_detection_model.pkl', help='Model file path')
    parser.add_argument('--data_file', type=str, required=True, help='HDF5 data file containing images for detection')
    parser.add_argument('--image_index', type=int, default=0, help='Index of the image to analyze in the dataset')
    parser.add_argument('--output', type=str, help='Path to save visualization output')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for feature extraction')
    parser.add_argument('--overlap', type=float, default=0.5, help='Window overlap ratio')
    parser.add_argument('--threshold', type=float, help='Custom threshold (overrides saved threshold)')
    parser.add_argument('--precision_mode', action='store_true', 
                      help='Enable precision mode (stricter detection criteria)')
    parser.add_argument('--adjustment_factor', type=float, default=1.0,
                      help='Adjust threshold by this factor (> 1.0 increases precision)')
    parser.add_argument('--position_aware', action='store_true',
                      help='Enable position-aware threshold adjustment')
    parser.add_argument('--strict_mode', action='store_true',
                      help='Enable strict mode for maximum precision')
    
    args = parser.parse_args()
    
    # Load model
    model, threshold, model_data = load_model(args.model)
    
    # Override threshold if specified
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using custom threshold: {threshold}")
    
    # Apply adjustment factor
    if args.adjustment_factor != 1.0:
        threshold = threshold * args.adjustment_factor
        print(f"Adjusted threshold: {threshold} (factor: {args.adjustment_factor})")
    
    # Additional strictness for strict mode
    if args.strict_mode:
        threshold *= 1.3
        print(f"Strict mode enabled, further adjusting threshold to: {threshold}")
    
    # Load image from HDF5 file
    print(f"Loading image {args.image_index} from {args.data_file}")
    try:
        with h5py.File(args.data_file, 'r') as f:
            if 'images' not in f:
                print(f"Error: No 'images' dataset found in {args.data_file}")
                return
            
            if args.image_index >= len(f['images']):
                print(f"Error: Image index {args.image_index} out of range (dataset has {len(f['images'])} images)")
                return
            
            image = f['images'][args.image_index]
            
            # Check if we have ground truth dead elements
            true_defects = None
            if 'dead_elements' in f:
                dead_elements = f['dead_elements'][args.image_index]
                true_defects = np.where(dead_elements > 0)[0].tolist()
    except Exception as e:
        print(f"Error loading HDF5 file: {e}")
        return
    
    # Detect defects with precision mode
    defect_positions = detect_defects_in_image(
        image, model, threshold, 
        window_size=args.window_size, 
        overlap=args.overlap,
        precision_mode=args.precision_mode or args.strict_mode
    )
    
    # Additional filtering in strict mode
    if args.strict_mode and defect_positions:
        # Keep only very high confidence detections
        high_conf_positions = []
        for pos, conf in defect_positions:
            if conf >= threshold * 1.1:
                high_conf_positions.append((pos, conf))
        
        # If filtering removed all detections but there were some before,
        # keep the single highest confidence detection
        if not high_conf_positions and defect_positions:
            best_position = max(defect_positions, key=lambda x: x[1])
            high_conf_positions = [best_position]
        
        defect_positions = high_conf_positions
    
    # Print results
    if len(defect_positions) > 0:
        print(f"Found {len(defect_positions)} defects:")
        for pos, conf in defect_positions:
            print(f"  Position {pos}: confidence {conf:.4f}")
    else:
        print("No defects detected.")
    
    # Compare with ground truth if available
    if true_defects is not None:
        print(f"\nGround truth has {len(true_defects)} defects at positions: {true_defects}")
        
        # Calculate accuracy metrics
        pred_positions = [pos for pos, _ in defect_positions]
        true_positives = len(set(pred_positions) & set(true_defects))
        false_positives = len(set(pred_positions) - set(true_defects))
        false_negatives = len(set(true_defects) - set(pred_positions))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # Create and save visualization
    # Convert image to 8-bit grayscale if needed
    if image.max() > 1.0:
        vis_image_base = normalize_image(image)
    else:
        vis_image_base = (image * 255).astype(np.uint8)
    
    vis_image = visualize_defects(vis_image_base, defect_positions)
    
    # Overlay ground truth if available
    if true_defects is not None:
        height = vis_image.shape[0]
        for pos in true_defects:
            # Draw green ground truth markers
            cv2.line(vis_image, (pos, 0), (pos, height//8), (0, 255, 0), 1)
            cv2.circle(vis_image, (pos, height//6), 3, (0, 255, 0), 1)
    
    if args.output:
        cv2.imwrite(args.output, vis_image)
        print(f"Visualization saved to {args.output}")
    else:
        # Display visualization
        cv2.imshow('Defect Detection', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()