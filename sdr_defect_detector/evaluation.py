import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix

def determine_optimal_threshold(model, X_val, y_val):
    """Find optimal threshold for converting probabilities to predictions"""
    # Get predicted probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    
    # Calculate F1 score for each threshold
    f1_scores = []
    for p, r in zip(precision, recall):
        if p + r == 0:
            f1 = 0
        else:
            f1 = 2 * (p * r) / (p + r)
        f1_scores.append(f1)
    
    # Find threshold with maximum F1 score
    best_idx = np.argmax(f1_scores[:-1])  # Exclude last point (corresponds to threshold=0)
    optimal_threshold = thresholds[best_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"At this threshold - Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}")
    
    return optimal_threshold

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
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(cm)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def evaluate_element_level_performance(model, threshold, test_images, true_defect_positions, preprocess_func, extract_features_func):
    """Evaluate model performance at element level"""
    from preprocessing import normalize_image, extract_roi, create_windows
    
    all_true_positions = []
    all_pred_positions = []
    
    for i, (image, true_positions) in enumerate(zip(test_images, true_defect_positions)):
        # Process image
        norm_image = normalize_image(image)
        roi = extract_roi(norm_image)
        
        # Create windows
        windows, positions = create_windows(roi)
        
        # Extract features
        window_features = [extract_features_func(window, norm_image, pos) for window, pos in zip(windows, positions)]
        
        # Get predictions
        probas = model.predict_proba(window_features)[:, 1]
        
        # Apply threshold
        predictions = probas >= threshold
        
        # Map to element positions (handling overlapping windows)
        element_probas = {}
        for (start, end), proba in zip(positions, probas):
            for pos in range(start, end):
                if pos not in element_probas or proba > element_probas[pos]:
                    element_probas[pos] = proba
        
        # Get predicted defect positions
        pred_positions = [pos for pos, prob in element_probas.items() if prob >= threshold]
        
        # Add to overall lists
        all_true_positions.extend([(i, pos) for pos in true_positions])
        all_pred_positions.extend([(i, pos) for pos in pred_positions])
    
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