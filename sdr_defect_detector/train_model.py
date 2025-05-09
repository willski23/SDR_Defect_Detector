import argparse
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Import from our modules
from preprocessing import normalize_image, extract_roi, create_windows
from feature_extraction import extract_window_features
from evaluation import determine_optimal_threshold, evaluate_model, evaluate_element_level_performance
from visualization import visualize_feature_importance

def save_model(model, threshold, metrics, feature_importance=None, filename='defect_detection_model.pkl'):
    """Save trained model and related information"""
    model_data = {
        'model': model,
        'threshold': threshold,
        'metrics': metrics,
        'feature_importance': feature_importance,
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

def custom_precision_weighted_f1(y_true, y_pred, beta=0.2):
    """Custom F-beta score with strong emphasis on precision"""
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    # F-beta score formula (beta < 1 emphasizes precision)
    # When beta=0.2, precision is weighted 25x more than recall
    if (prec + rec) > 0:
        return (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)
    return 0

def train_defect_detection_model(X_train, y_train, precision_focus=0.7):
    """Train and optimize defect detection model with focus on precision"""
    # Check for class imbalance
    class_counts = Counter(y_train)
    print(f"Class distribution: {class_counts}")
    
    # Calculate more aggressive class weights to reduce false positives
    # Higher weight for negative class reduces false positives
    weight_ratio = 2.0 + 3.0 * precision_focus  # More aggressive weighting
    class_weight = {
        0: weight_ratio,  # Much higher weight for negative class
        1: 1.0  # Normal weight for positive class
    }
    print(f"Using class weights: {class_weight}")
    
    # Handle class imbalance with limited SMOTE
    if class_counts[1] / sum(class_counts.values()) < 0.3:
        # Use very limited oversampling for high precision
        over = SMOTE(sampling_strategy=max(0.1, 0.3 - 0.2 * precision_focus), random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.35, random_state=42)
        
        steps = [('over', over), ('under', under)]
        pipeline = Pipeline(steps=steps)
        
        X_train, y_train = pipeline.fit_resample(X_train, y_train)
        print(f"After resampling: {Counter(y_train)}")
    
    # Model with extreme precision-focused parameters
    rf = RandomForestClassifier(
        n_estimators=800,         # More trees for better stability
        max_depth=12,             # Reduced to avoid overfitting
        min_samples_split=10,     # Increased to reduce false positives
        min_samples_leaf=8,       # Increased to reduce false positives
        max_features='sqrt',      # Reduces overfitting
        class_weight=class_weight,
        criterion='entropy',      # Better for imbalanced problems
        bootstrap=True,
        oob_score=True,           # Out-of-bag scoring
        n_jobs=-1,
        random_state=42
    )
    
    # Custom scorer for extreme precision focus
    beta = max(0.1, 0.3 - 0.2 * precision_focus)  # Adjust beta based on precision_focus
    
    def precision_focused_scorer(y_true, y_pred):
        return custom_precision_weighted_f1(y_true, y_pred, beta=beta)
    
    custom_scorer = make_scorer(precision_focused_scorer)
    
    # Hyperparameter optimization with more extreme options
    param_grid = {
        'n_estimators': [600, 800, 1000],
        'max_depth': [10, 12, 15],
        'min_samples_split': [8, 10, 15],
        'min_samples_leaf': [6, 8, 10]
    }
    
    # Use StratifiedKFold for imbalanced data
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring=custom_scorer, n_jobs=-1, verbose=1
    )
    
    # Train the model
    print("Training model with precision focus...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"OOB Score: {best_model.oob_score_:.4f}")
    
    # Feature importance analysis
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Top 10 features:")
    for f in range(min(10, len(importances))):
        print(f"{f+1}. Feature {indices[f]} ({importances[indices[f]]:.4f})")
    
    # Return the model and feature importance information
    feature_importance = {
        'values': importances,
        'indices': indices.tolist()
    }
    
    return best_model, feature_importance

def enhanced_calibration_for_precision(model, X_val, y_val, precision_focus=0.9):
    """Calibrate model probabilities with precision focus"""
    print("Performing enhanced probability calibration...")
    
    # Use isotonic regression for calibration
    calibrated_model = CalibratedClassifierCV(
        model, 
        method='isotonic',  
        cv='prefit'
    )
    
    # Fit the calibrator
    calibrated_model.fit(X_val, y_val)
    
    # Get calibrated probabilities
    y_proba = calibrated_model.predict_proba(X_val)[:, 1]
    
    # Analyze and adjust calibration curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10)
    
    # Create a visualizing plot
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Probability Calibration Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('calibration_curve.png')
    plt.close()
    
    print("Calibration curve saved to 'calibration_curve.png'")
    
    return calibrated_model

def find_precision_optimized_threshold(model, X_val, y_val, precision_focus=0.9):
    """Find threshold optimized for high precision"""
    from sklearn.metrics import precision_recall_curve
    
    # Get predicted probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    
    # Target minimum precision based on precision_focus
    # Higher precision_focus means higher target precision
    min_precision_target = 0.7 + (0.25 * precision_focus)
    print(f"Target minimum precision: {min_precision_target:.4f}")
    
    # Find the threshold that gives at least the target precision
    # while maximizing recall
    valid_indices = np.where(precision >= min_precision_target)[0]
    
    if len(valid_indices) > 0:
        # Among thresholds with acceptable precision, find the one with best recall
        best_idx = valid_indices[np.argmax(recall[valid_indices])]
        optimal_threshold = thresholds[best_idx - 1] if best_idx > 0 else 0.5
        
        print(f"Precision-optimized threshold: {optimal_threshold:.4f}")
        print(f"At this threshold - Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}")
    else:
        # If no threshold meets the precision requirement, choose a high threshold
        optimal_threshold = 0.9
        print(f"No threshold meets precision target. Using high threshold: {optimal_threshold:.4f}")
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, marker='.', linewidth=1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axhline(y=min_precision_target, color='r', linestyle='--', label=f'Target Precision ({min_precision_target:.2f})')
    
    # Mark the selected threshold
    for i, t in enumerate(thresholds):
        if abs(t - optimal_threshold) < 0.01:
            plt.plot(recall[i], precision[i], 'ro', markersize=8, 
                     label=f'Selected Threshold: {optimal_threshold:.2f}')
            break
    
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    print("Precision-recall curve saved to 'precision_recall_curve.png'")
    
    return optimal_threshold

def extract_features_with_validation(image, defects, window_size=8, overlap=0.5):
    """Extract features with validation for NaN and invalid values"""
    # Preprocess
    if image.max() > 1.0:
        norm_image = normalize_image(image)
    else:
        norm_image = (image * 255).astype(np.uint8)
        
    roi = extract_roi(norm_image)
    
    # Create windows
    windows, positions = create_windows(roi, window_size=window_size, overlap=overlap)
    
    features = []
    labels = []
    valid_indices = []
    
    for i, (window, (start, end)) in enumerate(zip(windows, positions)):
        # Extract features
        window_features = extract_window_features(window, norm_image, (start, end))
        
        # Validate features - check for NaN or inf
        if np.any(np.isnan(window_features)) or np.any(np.isinf(window_features)):
            # Skip this window
            continue
        
        features.append(window_features)
        
        # Determine label
        is_defective = any(start <= pos < end for pos in defects)
        labels.append(1 if is_defective else 0)
        valid_indices.append(i)
    
    return features, labels, valid_indices

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Precision-Focused Ultrasound Transducer Defect Detection Model')
    parser.add_argument('--data_file', type=str, required=True, help='HDF5 data file')
    parser.add_argument('--output', type=str, default='defect_detection_model.pkl', help='Output model filename')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for feature extraction')
    parser.add_argument('--overlap', type=float, default=0.5, help='Window overlap ratio')
    parser.add_argument('--precision_focus', type=float, default=0.9, 
                       help='Focus on precision (0-1, higher values prioritize precision over recall)')
    parser.add_argument('--save_vis', action='store_true', help='Save visualization of feature importance')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Load data from HDF5 file
    print(f"Loading dataset from HDF5 file: {args.data_file}...")
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
    print(f"Images with defects: {defect_count} ({defect_count/len(images)*100:.1f}%)")
    
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
    
    print(f"Training set: {len(train_images)} images ({sum(1 for d in train_defects if d)} with defects)")
    print(f"Validation set: {len(val_images)} images ({sum(1 for d in val_defects if d)} with defects)")
    print(f"Test set: {len(test_images)} images ({sum(1 for d in test_defects if d)} with defects)")
    
    # Process training images and extract features
    print("Extracting features from training set...")
    train_features = []
    train_labels = []
    
    for img_idx, (image, defects) in enumerate(zip(train_images, train_defects)):
        features, labels, _ = extract_features_with_validation(
            image, defects, 
            window_size=args.window_size, 
            overlap=args.overlap
        )
        train_features.extend(features)
        train_labels.extend(labels)
    
    # Convert to numpy arrays
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    
    print(f"Extracted {len(X_train)} valid feature vectors from training set")
    print(f"Positive samples: {np.sum(y_train)} ({np.sum(y_train)/len(y_train)*100:.2f}%)")
    
    # Train model
    print(f"Training model with precision_focus={args.precision_focus}...")
    model, feature_importance = train_defect_detection_model(
        X_train, y_train, 
        precision_focus=args.precision_focus
    )
    
    # Extract validation features
    print("Extracting features from validation set...")
    val_features = []
    val_labels = []
    
    for img_idx, (image, defects) in enumerate(zip(val_images, val_defects)):
        features, labels, _ = extract_features_with_validation(
            image, defects,
            window_size=args.window_size, 
            overlap=args.overlap
        )
        val_features.extend(features)
        val_labels.extend(labels)
    
    X_val = np.array(val_features)
    y_val = np.array(val_labels)
    
    print(f"Extracted {len(X_val)} valid feature vectors from validation set")
    
    # Calibrate model probabilities
    print("Calibrating model probabilities...")
    calibrated_model = enhanced_calibration_for_precision(
        model, X_val, y_val, 
        precision_focus=args.precision_focus
    )
    
    # Find precision-optimized threshold
    print("Finding precision-optimized threshold...")
    optimal_threshold = find_precision_optimized_threshold(
        calibrated_model, X_val, y_val, 
        precision_focus=args.precision_focus
    )
    
    # Evaluate on test set
    print("Extracting features from test set...")
    test_features = []
    test_labels = []
    
    for img_idx, (image, defects) in enumerate(zip(test_images, test_defects)):
        features, labels, _ = extract_features_with_validation(
            image, defects,
            window_size=args.window_size, 
            overlap=args.overlap
        )
        test_features.extend(features)
        test_labels.extend(labels)
    
    X_test = np.array(test_features)
    y_test = np.array(test_labels)
    
    print(f"Extracted {len(X_test)} valid feature vectors from test set")
    
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
    if args.save_vis:
        fig = visualize_feature_importance(model)
        fig.savefig('feature_importance.png')
        print("Feature importance visualization saved to 'feature_importance.png'")
    
    # Save calibrated model
    metrics = {
        'window_level': window_metrics,
        'element_level': element_metrics
    }
    save_model(calibrated_model, optimal_threshold, metrics, feature_importance, args.output)
    
    print(f"Training complete! Model saved to {args.output}")
    print(f"Element-level precision: {element_metrics['precision']:.4f}")
    print(f"Element-level recall: {element_metrics['recall']:.4f}")
    print(f"Element-level F1 score: {element_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()