import argparse
import numpy as np
import h5py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing import normalize_image
from element_features import simple_element_features

def save_model(model, threshold, metrics, filename='element_classifier.pkl'):
    """Save trained model and metrics"""
    model_data = {
        'model': model,
        'threshold': threshold,
        'metrics': metrics,
        'timestamp': np.datetime64('now')
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Element classifier saved to {filename}")

def train_element_classifier(images, true_defect_positions, extreme_precision=True):
    """Train a classifier specifically for single elements using simple features"""
    # Collect training data
    print("Collecting element training data...")
    positive_features = []
    negative_features = []
    
    for i, (image, defects) in enumerate(tqdm(zip(images, true_defect_positions), total=len(images))):
        # Skip images without defects
        if not defects:
            continue
            
        # Ensure image is normalized
        if image.max() > 1.0:
            norm_image = normalize_image(image)
        else:
            norm_image = (image * 255).astype(np.uint8)
            
        # Extract features for true defects (all positive examples)
        for defect_pos in defects:
            if defect_pos < 0 or defect_pos >= norm_image.shape[1]:
                continue  # Skip invalid positions
                
            try:
                features = simple_element_features(norm_image, defect_pos)
                # Skip if all features are zero or contain invalid values
                if not np.all(features == 0) and not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                    positive_features.append(features)
            except Exception as e:
                print(f"Error extracting positive features at position {defect_pos}: {e}")
        
        # Sample non-defect elements (negative examples)
        non_defect_positions = [p for p in range(norm_image.shape[1]) 
                              if p not in defects and p < norm_image.shape[1]]
        
        # For high precision, use more negative samples
        if extreme_precision:
            # Use 5x more negative samples for extreme precision
            sample_ratio = 5
        else:
            # Use balanced samples
            sample_ratio = 1
            
        # Sample negative examples - ensure at least some samples even if no defects
        max_samples = min(len(non_defect_positions), 
                        max(10, sample_ratio * max(1, len(defects))))
        
        if max_samples > 0 and len(non_defect_positions) > 0:
            sampled_negative = np.random.choice(non_defect_positions, max_samples, replace=False)
            
            for neg_pos in sampled_negative:
                try:
                    features = simple_element_features(norm_image, neg_pos)
                    # Skip if all features are zero or contain invalid values
                    if not np.all(features == 0) and not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                        negative_features.append(features)
                except Exception as e:
                    pass  # Silently skip errors in negative samples
    
    # Prepare training data
    if len(positive_features) < 10 or len(negative_features) < 10:
        print(f"Error: Not enough samples for training. Found {len(positive_features)} positive and {len(negative_features)} negative samples.")
        return None, None
        
    X_positive = np.array(positive_features)
    X_negative = np.array(negative_features)
    
    print(f"Collected {len(X_positive)} positive and {len(X_negative)} negative samples")
    
    X = np.vstack([X_positive, X_negative])
    y = np.hstack([np.ones(len(X_positive)), np.zeros(len(X_negative))])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Set class weights for precision focus
    if extreme_precision:
        # Very aggressive class weights for extreme precision
        class_weight = {0: 20, 1: 1}
    else:
        # More balanced but still precision-focused
        class_weight = {0: 5, 1: 1}
    
    print(f"Using class weights: {class_weight}")
    
    # Train the classifier
    print("Training element classifier...")
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        criterion='entropy',
        n_jobs=-1,
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    # Find optimal threshold for high precision
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calculate precision at different thresholds
    thresholds = np.linspace(0.1, 0.9, 9)
    best_threshold = 0.5
    best_f1 = 0
    best_precision = 0
    
    print("\nThreshold optimization:")
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"Threshold {threshold:.1f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # For extreme precision, prioritize precision score
        if extreme_precision:
            if precision > best_precision or (precision >= best_precision and f1 > best_f1):
                best_precision = precision
                best_threshold = threshold
                best_f1 = f1
        else:
            # For balanced approach, prioritize F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
    
    # Final evaluation with best threshold
    print(f"\nSelected threshold: {best_threshold:.2f}")
    final_y_pred = (y_proba >= best_threshold).astype(int)
    
    final_precision = precision_score(y_test, final_y_pred, zero_division=0)
    final_recall = recall_score(y_test, final_y_pred, zero_division=0)
    final_f1 = f1_score(y_test, final_y_pred, zero_division=0)
    
    print(f"Final metrics - Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, F1: {final_f1:.4f}")
    
    # Save feature importance visualization
    feature_names = [
        "Mean Intensity", "Std Deviation", "Median Intensity", "Max Intensity", "Min Intensity",
        "Left Difference", "Right Difference", "Edge Distance", "Top/Bottom Ratio", "Column Contrast"
    ]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(X.shape[1]), importances[indices])
    plt.yticks(range(X.shape[1]), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Element Classifier Feature Importance')
    plt.tight_layout()
    plt.savefig('element_feature_importance.png')
    plt.close()
    
    # Return the classifier and optimal threshold
    return clf, best_threshold

def main():
    parser = argparse.ArgumentParser(description='Train Element-Level Classifier for Defect Detection')
    parser.add_argument('--data_file', type=str, required=True, help='HDF5 data file')
    parser.add_argument('--output', type=str, default='element_classifier.pkl', help='Output model file')
    parser.add_argument('--extreme_precision', action='store_true', help='Optimize for extreme precision')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.data_file}")
    with h5py.File(args.data_file, 'r') as f:
        images = f['images'][:]
        
        # Load defect positions
        true_defect_positions = []
        for elements in f['dead_elements'][:]:
            positions = np.where(elements > 0)[0].tolist()
            true_defect_positions.append(positions)
    
    # Train the classifier
    clf, threshold = train_element_classifier(
        images, 
        true_defect_positions,
        extreme_precision=args.extreme_precision
    )
    
    if clf is not None:
        # Save the model
        metrics = {
            'precision_focus': args.extreme_precision
        }
        save_model(clf, threshold, metrics, args.output)

if __name__ == "__main__":
    main()