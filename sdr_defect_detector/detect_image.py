import argparse
import numpy as np
import h5py
import cv2
import pickle
import os

from preprocessing import normalize_image, extract_roi, create_windows
from feature_extraction import extract_window_features
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

def detect_defects_in_image(image, model, threshold, window_size=8, overlap=0.5):
    """Detect defects in an image"""
    # Ensure image is properly normalized
    if image.max() > 1.0:
        norm_image = normalize_image(image)
    else:
        norm_image = (image * 255).astype(np.uint8)
    
    # Extract ROI
    roi = extract_roi(norm_image)
    
    # Create windows
    windows, positions = create_windows(roi, window_size=window_size, overlap=overlap)
    
    # Extract features
    features = []
    for window, pos in zip(windows, positions):
        window_features = extract_window_features(window, norm_image, pos)
        features.append(window_features)
    
    # Get predictions
    if len(features) > 0:
        X = np.array(features)
        probas = model.predict_proba(X)[:, 1]
        
        # Find defective elements
        defect_positions = []
        element_probas = {}
        
        # Handle overlapping windows
        for (start, end), proba in zip(positions, probas):
            for pos in range(start, end):
                if pos not in element_probas or proba > element_probas[pos]:
                    element_probas[pos] = proba
        
        # Get positions with confidence above threshold
        for pos, conf in element_probas.items():
            if conf >= threshold:
                defect_positions.append((pos, conf))
        
        # Sort by position
        defect_positions.sort(key=lambda x: x[0])
        
        return defect_positions
    
    return []

def main():
    parser = argparse.ArgumentParser(description='Detect Defects in Ultrasound Transducer Image')
    parser.add_argument('--model', type=str, default='defect_detection_model.pkl', help='Model file path')
    parser.add_argument('--data_file', type=str, required=True, help='HDF5 data file containing images for detection')
    parser.add_argument('--image_index', type=int, default=0, help='Index of the image to analyze in the dataset')
    parser.add_argument('--output', type=str, help='Path to save visualization output')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for feature extraction')
    parser.add_argument('--overlap', type=float, default=0.5, help='Window overlap ratio')
    parser.add_argument('--threshold', type=float, help='Custom threshold (overrides saved threshold)')
    
    args = parser.parse_args()
    
    # Load model - CHANGE PATH TO YOUR MODEL FILE
    #model_path = "C:\\Users\\wbszy\\code_projects\\Soundcheck\\defect_detection_model.pkl"
    model, threshold, model_data = load_model(args.model)
    
    # Override threshold if specified
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using custom threshold: {threshold}")
    
    # Load image from HDF5 file
    print(f"Loading image {args.image_index} from {args.data_file}")
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
    
    # Detect defects
    defect_positions = detect_defects_in_image(
        image, model, threshold, 
        window_size=args.window_size, 
        overlap=args.overlap
    )
    
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