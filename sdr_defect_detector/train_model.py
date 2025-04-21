import argparse
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter

# Import from our modules
from preprocessing import normalize_image, extract_roi, create_windows
from feature_extraction import extract_window_features
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

def train_defect_detection_model(X_train, y_train, precision_focus=0.7):
    """Train and optimize defect detection model with focus on precision"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    
    # Check for class imbalance
    class_counts = Counter(y_train)
    print(f"Class distribution: {class_counts}")
    
    # Calculate class weights to emphasize precision
    # Higher weight for negative class reduces false positives
    weight_ratio = 1.0 + precision_focus  # Adjust negative class weight
    class_weight = {
        0: weight_ratio,  # Increased weight for negative class
        1: 1.0  # Normal weight for positive class
    }
    
    # Handle class imbalance with SMOTE but slightly under-sample the minority class
    if class_counts[1] / sum(class_counts.values()) < 0.3:
        # Create a pipeline with both over and under sampling
        over = SMOTE(sampling_strategy=0.2, random_state=42)  # Less aggressive SMOTE
        under = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
        
        steps = [('over', over), ('under', under)]
        pipeline = Pipeline(steps=steps)
        
        X_train, y_train = pipeline.fit_resample(X_train, y_train)
        print(f"After resampling: {Counter(y_train)}")
    
    # Model with precision-focused parameters
    rf = RandomForestClassifier(
        n_estimators=500,  # More trees for better generalization
        max_depth=15,      # Reduced to avoid overfitting
        min_samples_split=8,  # Increased to reduce false positives
        min_samples_leaf=4,   # Increased to reduce false positives
        class_weight=class_weight,
        criterion='gini',  # Try 'entropy' as alternative
        n_jobs=-1,
        random_state=42
    )
    
    # Custom scorer that weights precision more than recall
    def precision_focused_f1(y_true, y_pred):
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        # Weighted harmonic mean with emphasis on precision
        beta = 0.5  # Beta < 1 gives more weight to precision
        return (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec) if (prec + rec) > 0 else 0
    
    custom_scorer = make_scorer(precision_focused_f1)
    
    # Hyperparameter optimization
    param_grid = {
        'n_estimators': [400, 500, 600],
        'max_depth': [12, 15, 18],
        'min_samples_split': [6, 8, 10]
    }
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring=custom_scorer, n_jobs=-1
    )
    
    # Train the model
    print("Training model with precision focus...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Feature importance analysis
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Top 10 features:")
    for f in range(min(10, len(importances))):
        print(f"{f+1}. Feature {indices[f]} ({importances[indices[f]]:.4f})")
    
    return best_model

def calibrate_probabilities(model, X_val, y_val):
    """Calibrate model probability outputs for more reliable thresholding"""
    from sklearn.calibration import CalibratedClassifierCV
    
    # Calibrate model probabilities
    calibrated_model = CalibratedClassifierCV(
        model, 
        method='isotonic',  # or 'sigmoid'
        cv='prefit'  # Use prefit model
    )
    
    # Fit the calibrator
    calibrated_model.fit(X_val, y_val)
    
    return calibrated_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Ultrasound Transducer Defect Detection Model')
    parser.add_argument('--data_file', type=str, default='dataset.h5', help='HDF5 data file')
    parser.add_argument('--output', type=str, default='defect_detection_model.pkl', help='Output model filename')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for feature extraction')
    parser.add_argument('--overlap', type=float, default=0.5, help='Window overlap ratio')
    parser.add_argument('--precision_focus', type=float, default=0.7, 
                       help='Focus on precision (0-1, higher values prioritize precision over recall)')
    
    args = parser.parse_args()
    
    # Load data from HDF5 file
    print("Loading dataset from HDF5 file...")
    images, dead_elements, filenames = load_data_from_hdf5(args.data_file)
    
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
        # Preprocess
        if image.max() > 1.0:
            norm_image = normalize_image(image)
        else:
            norm_image = (image * 255).astype(np.uint8)
            
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
    model = train_defect_detection_model(X_train, y_train, precision_focus=args.precision_focus)
    
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
    
    # Calibrate model probabilities
    print("Calibrating model probabilities...")
    calibrated_model = calibrate_probabilities(model, X_val, y_val)
    
    # Re-determine optimal threshold with calibrated model
    print("Re-determining optimal threshold with calibrated model...")
    optimal_threshold = determine_optimal_threshold(calibrated_model, X_val, y_val)
    
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
    window_metrics = evaluate_model(calibrated_model, optimal_threshold, X_test, y_test)
    
    # Evaluate element-level performance
    print("Evaluating element-level performance...")
    element_metrics = evaluate_element_level_performance(
        calibrated_model, 
        optimal_threshold, 
        test_images, 
        test_defects,
        lambda x: normalize_image(x) if x.max() > 1.0 else (x * 255).astype(np.uint8), 
        extract_window_features
    )
    
    # Visualize feature importance
    fig = visualize_feature_importance(calibrated_model)
    fig.savefig('feature_importance.png')
    
    # Save calibrated model
    metrics = {
        'window_level': window_metrics,
        'element_level': element_metrics
    }
    save_model(calibrated_model, optimal_threshold, metrics, args.output)
    
    print("Training complete!")

if __name__ == "__main__":
    main()