import argparse
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import cv2
import os
import pickle

# Import from our modules
from preprocessing import normalize_image, extract_roi, create_windows
from feature_extraction import extract_window_features
from model import train_defect_detection_model
from evaluation import determine_optimal_threshold, evaluate_model, evaluate_element_level_performance
from visualization import visualize_feature_importance

def save_model(model, threshold, metrics, filename='defect_detection_model.pkl'):
    """Save trained model and related information"""
    model_data = {
        'model': model,
        'threshold': threshold,
        'metrics': metrics,
        'timestamp': np.datetime64('now')
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filename}")

def load_data_from_hdf5(filepath):
    """Load image data and dead element information from HDF5 file"""
    with h5py.File(filepath, 'r') as f:
        # Load images
        images = f['images'][:]
        
        # Load dead elements positions
        dead_elements = f['dead_elements'][:]
        
        # Optionally load filenames if needed
        filenames = f['filenames'][:] if 'filenames' in f else None
        
    return images, dead_elements, filenames

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Ultrasound Transducer Defect Detection Model')
    parser.add_argument('--data_file', type=str, default='dataset.h5', help='HDF5 data file')
    parser.add_argument('--output', type=str, default='defect_detection_model.pkl', help='Output model filename')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for feature extraction')
    parser.add_argument('--overlap', type=float, default=0.5, help='Window overlap ratio')
    
    args = parser.parse_args()
    
    # Load data from HDF5 file
    print("Loading dataset from HDF5 file...")
    dataset_path = "C:\\Users\\wbszy\\code_projects\\Soundcheck\\data\\processed\\dataset.h5"
    images, dead_elements, filenames = load_data_from_hdf5(dataset_path)
    
    print(f"Dataset: {len(images)} images")
    
    # Convert dead_elements to lists of defect positions for each image
    defect_labels = []
    for elements in dead_elements:
        # Add non-zero indices as defect positions
        positions = np.where(elements > 0)[0].tolist()
        defect_labels.append(positions)
    
    # Count images with defects
    defect_count = sum(1 for labels in defect_labels if len(labels) > 0)
    print(f"Images with defects: {defect_count}")
    
    # Split into train, validation, test sets
    has_defect = [1 if len(labels) > 0 else 0 for labels in defect_labels]
    
    # First split for test set
    train_val_indices, test_indices = train_test_split(
        np.arange(len(images)), 
        test_size=0.2, 
        random_state=42,
        stratify=has_defect
    )
    
    # Further split into train and validation
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.25,
        random_state=42,
        stratify=[has_defect[i] for i in train_val_indices]
    )
    
    # Extract datasets
    train_images = [images[i] for i in train_indices]
    train_defects = [defect_labels[i] for i in train_indices]
    
    val_images = [images[i] for i in val_indices]
    val_defects = [defect_labels[i] for i in val_indices]
    
    test_images = [images[i] for i in test_indices]
    test_defects = [defect_labels[i] for i in test_indices]
    
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    print(f"Test set: {len(test_images)} images")
    
    # Process training images and extract features
    print("Extracting features from training set...")
    train_features = []
    train_labels = []
    
    for img_idx, (image, defects) in enumerate(zip(train_images, train_defects)):
        # Note: HDF5 data may already be normalized, but we'll ensure consistency
        if image.max() > 1.0:  # Check if normalization is needed
            norm_image = normalize_image(image)
        else:
            norm_image = (image * 255).astype(np.uint8)  # Scale to 0-255 range
        
        roi = extract_roi(norm_image)
        
        # Create windows
        windows, positions = create_windows(roi, window_size=args.window_size, overlap=args.overlap)
        
        for window, (start, end) in zip(windows, positions):
            # Extract features
            features = extract_window_features(window, norm_image, (start, end))
            train_features.append(features)
            
            # Determine label
            is_defective = any(start <= pos < end for pos in defects)
            train_labels.append(1 if is_defective else 0)
    
    # Convert to numpy arrays
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    
    # Train model
    print(f"Training model on {len(X_train)} windows...")
    model = train_defect_detection_model(X_train, y_train)
    
    # Extract validation features
    print("Extracting features from validation set...")
    val_features = []
    val_labels = []
    
    for img_idx, (image, defects) in enumerate(zip(val_images, val_defects)):
        if image.max() > 1.0:
            norm_image = normalize_image(image)
        else:
            norm_image = (image * 255).astype(np.uint8)
            
        roi = extract_roi(norm_image)
        
        windows, positions = create_windows(roi, window_size=args.window_size, overlap=args.overlap)
        
        for window, (start, end) in zip(windows, positions):
            features = extract_window_features(window, norm_image, (start, end))
            val_features.append(features)
            
            is_defective = any(start <= pos < end for pos in defects)
            val_labels.append(1 if is_defective else 0)
    
    X_val = np.array(val_features)
    y_val = np.array(val_labels)
    
    # Determine optimal threshold
    print("Determining optimal threshold...")
    optimal_threshold = determine_optimal_threshold(model, X_val, y_val)
    
    # Evaluate on test set
    print("Extracting features from test set...")
    test_features = []
    test_labels = []
    test_window_map = []
    
    for img_idx, (image, defects) in enumerate(zip(test_images, test_defects)):
        if image.max() > 1.0:
            norm_image = normalize_image(image)
        else:
            norm_image = (image * 255).astype(np.uint8)
            
        roi = extract_roi(norm_image)
        
        windows, positions = create_windows(roi, window_size=args.window_size, overlap=args.overlap)
        
        for window, (start, end) in zip(windows, positions):
            features = extract_window_features(window, norm_image, (start, end))
            test_features.append(features)
            
            is_defective = any(start <= pos < end for pos in defects)
            test_labels.append(1 if is_defective else 0)
            
            test_window_map.append((img_idx, start, end))
    
    X_test = np.array(test_features)
    y_test = np.array(test_labels)
    
    # Evaluate window-level performance
    print("Evaluating window-level performance...")
    window_metrics = evaluate_model(model, optimal_threshold, X_test, y_test)
    
    # Evaluate element-level performance
    print("Evaluating element-level performance...")
    element_metrics = evaluate_element_level_performance(
        model, 
        optimal_threshold, 
        test_images, 
        test_defects,
        lambda x: normalize_image(x) if x.max() > 1.0 else (x * 255).astype(np.uint8), 
        extract_window_features
    )
    
    # Visualize feature importance
    fig = visualize_feature_importance(model)
    fig.savefig('feature_importance.png')
    
    # Save model
    metrics = {
        'window_level': window_metrics,
        'element_level': element_metrics
    }
    save_model(model, optimal_threshold, metrics, args.output)
    
    print("Training complete!")

if __name__ == "__main__":
    main()