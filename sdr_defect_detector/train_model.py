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
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, precision_recall_curve

# Import from our modules
from preprocessing import normalize_image, extract_roi, create_windows
from feature_extraction import extract_window_features, safe_correlation
from visualization import visualize_feature_importance, visualize_precision_recall_curve

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

def train_defect_detection_model(X_train, y_train, precision_focus=0.95):
    """Train model with extreme precision focus"""
    # Prepare for highly imbalanced data
    class_counts = Counter(y_train)
    print(f"Class distribution: {class_counts}")
    
    # Set very aggressive class weights for precision
    weight_ratio = 10.0 * precision_focus  # Much higher weight
    class_weight = {
        0: weight_ratio,  # Very high weight for negative class
        1: 1.0            # Normal weight for positive class
    }
    print(f"Using class weights: {class_weight}")
    
    # Use more trees and stricter parameters
    rf = RandomForestClassifier(
        n_estimators=1500,        # More trees for stability
        max_depth=8,              # Reduce depth to prevent overfitting
        min_samples_split=25,     # Higher to reduce false positives
        min_samples_leaf=15,      # Higher to reduce false positives
        max_features='sqrt',      # Better for imbalanced data
        class_weight=class_weight,
        criterion='entropy',      # Better for imbalanced data
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )
    
    # Use stratified k-fold for reliable validation
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Custom F-beta with extreme precision preference (beta=0.1)
    def precision_focused_score(y_true, y_pred):
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        beta = 0.1  # Very strong precision emphasis
        if (prec + rec) > 0:
            return (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)
        return 0
    
    custom_scorer = make_scorer(precision_focused_score)
    
    # Hyperparameter optimization with more extreme values
    param_grid = {
        'n_estimators': [1200, 1500, 1800],
        'min_samples_split': [20, 25, 30],
        'min_samples_leaf': [10, 15, 20],
    }
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring=custom_scorer, n_jobs=-1, verbose=1
    )
    
    # Train the model
    print("Training model with extreme precision focus...")
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
    # Get predicted probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    
    # Target minimum precision based on precision_focus
    # Higher precision_focus means higher target precision
    min_precision_target = 0.85 + (0.14 * precision_focus)
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
        optimal_threshold = 0.95
        print(f"No threshold meets precision target. Using high threshold: {optimal_threshold:.4f}")
    
    # Plot precision-recall curve
    fig = visualize_precision_recall_curve(precision, recall, thresholds, threshold=optimal_threshold)
    fig.savefig('precision_recall_curve.png')
    plt.close()
    
    print("Precision-recall curve saved to 'precision_recall_curve.png'")
    
    return optimal_threshold

def extract_features_with_validation(image, defects, window_size=8, overlap=0.5):
    """Extract features with improved validation for NaN and invalid values"""
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
        try:
            # Extract features
            window_features = extract_window_features(window, norm_image, (start, end))
            
            # Validate features - check for NaN or inf
            window_features = np.array(window_features)
            if np.any(np.isnan(window_features)) or np.any(np.isinf(window_features)):
                # Replace NaN/inf with zeros
                window_features[np.isnan(window_features)] = 0
                window_features[np.isinf(window_features)] = 0
            
            features.append(window_features)
            
            # Determine label
            is_defective = any(start <= pos < end for pos in defects)
            labels.append(1 if is_defective else 0)
            valid_indices.append(i)
        except Exception as e:
            # Skip this window
            continue
    
    return features, labels, valid_indices

def evaluate_model(model, threshold, X_test, y_test):
    """Evaluate model performance with the optimal threshold"""
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Test set performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_element_level_performance(model, threshold, test_images, true_defect_positions):
    """Evaluate model performance at element level"""
    all_true_positions = []
    all_pred_positions = []
    
    for i, (image, true_positions) in enumerate(zip(test_images, true_defect_positions)):
        # Process image
        if image.max() > 1.0:
            norm_image = normalize_image(image)
        else:
            norm_image = (image * 255).astype(np.uint8)
            
        roi = extract_roi(norm_image)
        
        # Create windows
        windows, positions = create_windows(roi, window_size=8, overlap=0.5)
        
        # Extract features
        features = []
        valid_positions = []
        
        for window, pos in zip(windows, positions):
            try:
                window_features = extract_window_features(window, norm_image, pos)
                
                # Validate features
                window_features = np.array(window_features)
                if not np.any(np.isnan(window_features)) and not np.any(np.isinf(window_features)):
                    features.append(window_features)
                    valid_positions.append(pos)
                else:
                    # Fix invalid values
                    window_features[np.isnan(window_features)] = 0
                    window_features[np.isinf(window_features)] = 0
                    features.append(window_features)
                    valid_positions.append(pos)
            except Exception as e:
                # Skip problematic features
                continue
        
        # Get predictions if we have features
        if len(features) > 0:
            X = np.array(features)
            
            try:
                # Get probability predictions
                probas = model.predict_proba(X)[:, 1]
                
                # Apply threshold and handle overlapping windows
                element_probas = {}
                for (start, end), proba in zip(valid_positions, probas):
                    for pos in range(start, end):
                        if pos not in element_probas or proba > element_probas[pos]:
                            element_probas[pos] = proba
                
                # Get predicted defect positions
                pred_positions = [pos for pos, prob in element_probas.items() if prob >= threshold]
                
                # Add to overall lists
                all_true_positions.extend([(i, pos) for pos in true_positions])
                all_pred_positions.extend([(i, pos) for pos in pred_positions])
            except Exception as e:
                print(f"Error evaluating image {i}: {e}")
                continue
    
    # Calculate element-level metrics
    true_positives = len(set(all_true_positions) & set(all_pred_positions))
    false_positives = len(set(all_pred_positions) - set(all_true_positions))
    false_negatives = len(set(all_true_positions) - set(all_pred_positions))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Element-level metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Precision-Focused Ultrasound Transducer Defect Detection Model')
    parser.add_argument('--data_file', type=str, required=True, help='HDF5 data file')
    parser.add_argument('--output', type=str, default='defect_detection_model.pkl', help='Output model filename')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for feature extraction')
    parser.add_argument('--overlap', type=float, default=0.5, help='Window overlap ratio')
    parser.add_argument('--precision_focus', type=float, default=0.95, 
                       help='Focus on precision (0-1, higher values prioritize precision over recall)')
    parser.add_argument('--save_vis', action='store_true', help='Save visualization of feature importance')
    parser.add_argument('--extreme_precision', action='store_true', help='Use extreme precision settings')
    
    args = parser.parse_args()
    
    # Adjust precision focus if extreme mode requested
    if args.extreme_precision:
        precision_focus = 0.98
        print(f"Using extreme precision settings (precision_focus = {precision_focus})")
    else:
        precision_focus = args.precision_focus
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Load data from HDF5 file
    print(f"Loading dataset from HDF5 file: {args.data_file}...")
    try:
        with h5py.File(args.data_file, 'r') as f:
            # Load images
            images = f['images'][:]
            
            # Load dead elements
            if 'dead_elements' in f:
                dead_elements = f['dead_elements'][:]
            else:
                print("Error: No 'dead_elements' dataset found in HDF5 file")
                return
    except Exception as e:
        print(f"Error loading HDF5 file: {e}")
        return
    
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
    print(f"Training model with precision_focus={precision_focus}...")
    model, feature_importance = train_defect_detection_model(
        X_train, y_train, 
        precision_focus=precision_focus
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
        precision_focus=precision_focus
    )
    
    # Find precision-optimized threshold
    print("Finding precision-optimized threshold...")
    optimal_threshold = find_precision_optimized_threshold(
        calibrated_model, X_val, y_val, 
        precision_focus=precision_focus
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
        test_defects
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