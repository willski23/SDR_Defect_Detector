import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix

def determine_optimal_threshold(model, X_val, y_val):
    """Find optimal threshold for converting probabilities to predictions with improved precision focus"""
    # Get predicted probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    
    # Calculate F1 and F-beta scores for each threshold
    f1_scores = []
    f_beta_scores = [] # Using beta=0.5 to prioritize precision
    beta = 0.5
    
    for p, r in zip(precision, recall):
        if p + r == 0:
            f1 = 0
            f_beta = 0
        else:
            f1 = 2 * (p * r) / (p + r)
            f_beta = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)
        f1_scores.append(f1)
        f_beta_scores.append(f_beta)
    
    # Find threshold with maximum F1 score
    best_f1_idx = np.argmax(f1_scores[:-1])  # Exclude last point (corresponds to threshold=0)
    f1_threshold = thresholds[best_f1_idx]
    
    # Find threshold with maximum F-beta score
    best_f_beta_idx = np.argmax(f_beta_scores[:-1])
    f_beta_threshold = thresholds[best_f_beta_idx]
    
    # Find a precision-focused threshold (targeting >90% precision)
    target_precision = 0.9
    valid_indices = np.where(precision[:-1] >= target_precision)[0]
    
    if len(valid_indices) > 0:
        # Find highest recall among thresholds with good precision
        best_high_prec_idx = valid_indices[np.argmax(recall[valid_indices])]
        high_prec_threshold = thresholds[best_high_prec_idx]
        
        print(f"High precision threshold: {high_prec_threshold:.4f}")
        print(f"At this threshold - Precision: {precision[best_high_prec_idx]:.4f}, " 
              f"Recall: {recall[best_high_prec_idx]:.4f}, F1: {f1_scores[best_high_prec_idx]:.4f}")
    else:
        # If no threshold meets precision target, use F-beta threshold
        high_prec_threshold = f_beta_threshold
        print("Could not find threshold with precision >= 90%, using F-beta threshold")
    
    print(f"F1 optimal threshold: {f1_threshold:.4f}")
    print(f"At this threshold - Precision: {precision[best_f1_idx]:.4f}, "
          f"Recall: {recall[best_f1_idx]:.4f}, F1: {f1_scores[best_f1_idx]:.4f}")
    
    print(f"F-beta optimal threshold: {f_beta_threshold:.4f}")
    print(f"At this threshold - Precision: {precision[best_f_beta_idx]:.4f}, "
          f"Recall: {recall[best_f_beta_idx]:.4f}, F-beta: {f_beta_scores[best_f_beta_idx]:.4f}")
    
    # Plot precision-recall curve
    plt.figure(figsize=(12, 8))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.axvline(x=recall[best_f1_idx], color='g', linestyle='--', label=f'F1 Threshold: {f1_threshold:.3f}')
    plt.axvline(x=recall[best_f_beta_idx], color='r', linestyle='--', label=f'F-beta Threshold: {f_beta_threshold:.3f}')
    
    if len(valid_indices) > 0:
        plt.axvline(x=recall[best_high_prec_idx], color='purple', linestyle='--', 
                   label=f'High Precision Threshold: {high_prec_threshold:.3f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('threshold_selection.png')
    plt.close()
    
    # For precision-focused applications, return the high precision threshold
    return high_prec_threshold

def evaluate_model(model, threshold, X_test, y_test):
    """Evaluate model performance with the optimal threshold"""
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Test set performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(cm)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    print(f"Specificity: {specificity:.4f}")
    print(f"Negative Predictive Value: {npv:.4f}")
    
    # Calculate F-beta score (beta = 0.5 prioritizes precision)
    beta = 0.5
    if precision + recall > 0:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    else:
        f_beta = 0
        
    print(f"F-beta Score (beta=0.5): {f_beta:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'npv': npv,
        'f_beta': f_beta,
        'confusion_matrix': cm
    }

def evaluate_element_level_performance(model, threshold, test_images, true_defect_positions):
    """Evaluate model performance at element level"""
    from preprocessing import normalize_image, extract_roi, create_windows
    from feature_extraction import extract_window_features
    
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
        
        # Extract features with improved error handling
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
    
    # True negatives are hard to define precisely at element level,
    # but we can estimate using total positions - (TP + FP + FN)
    total_positions = 0
    for img in test_images:
        if img is not None and hasattr(img, 'shape') and len(img.shape) >= 2:
            total_positions += img.shape[1]
            
    true_negatives = max(0, total_positions - (true_positives + false_positives + false_negatives))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate additional metrics
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # Calculate F-beta score (beta = 0.5 prioritizes precision)
    beta = 0.5
    if precision + recall > 0:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    else:
        f_beta = 0
    
    print(f"Element-level metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F-beta (β=0.5): {f_beta:.4f}")
    print(f"True Positives: {true_positives}, False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}, Est. True Negatives: {true_negatives}")
    
    # Create precision-recall visualization
    plt.figure(figsize=(10, 8))
    plt.scatter([recall], [precision], color='red', marker='o', s=100)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Element-Level Precision vs Recall')
    plt.grid(alpha=0.3)
    
    # Add F1 curves
    f1_curves = np.linspace(0.1, 0.9, 9)
    for f in f1_curves:
        x = np.linspace(0.01, 0.99, 100)
        y = (f * x) / (2 * x - f)
        mask = (y >= 0) & (y <= 1)
        plt.plot(x[mask], y[mask], 'k--', alpha=0.3)
        # Add text label at midpoint
        midpoint = len(x[mask]) // 2
        if midpoint > 0:
            plt.annotate(f'F1={f:.1f}', 
                        (x[mask][midpoint], y[mask][midpoint]),
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=8)
    
    plt.savefig('element_level_metrics.png')
    plt.close()
    
    # Create confusion metrics visualization
    cm = np.array([
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Element-Level Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Normal', 'Defect'])
    plt.yticks([0, 1], ['Normal', 'Defect'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('element_level_confusion.png')
    plt.close()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f_beta': f_beta,
        'specificity': specificity,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }

def analyze_feature_importance(model, feature_names=None):
    """Analyze and visualize feature importance from a trained model"""
    # For calibrated models, access the base estimator
    if hasattr(model, 'base_estimator'):
        # For calibrated models, get the first calibrated estimator
        if hasattr(model, 'calibrated_classifiers_'):
            base_model = model.calibrated_classifiers_[0].base_estimator
        else:
            base_model = model.base_estimator
    else:
        base_model = model
    
    # Ensure model has feature_importances_ attribute
    if not hasattr(base_model, 'feature_importances_'):
        print("Model doesn't expose feature importances")
        return None
    
    # Get feature importances
    importances = base_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    # Top important features
    n_features = min(20, len(importances))
    
    # Print importance ranking
    print("Top 20 most important features:")
    for i in range(n_features):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 10))
    plt.barh(range(n_features), importances[indices[:n_features]], align='center')
    plt.yticks(range(n_features), [feature_names[i] for i in indices[:n_features]])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Ranking')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Also create a pie chart for top 10 features
    top_n = min(10, len(importances))
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Add percentage to names
    total_importance = sum(top_importances)
    pie_labels = [f"{name} ({100*imp/total_importance:.1f}%)" for name, imp in zip(top_names, top_importances)]
    
    plt.figure(figsize=(10, 10))
    plt.pie(top_importances, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Top 10 Feature Importance Distribution')
    plt.savefig('feature_importance_pie.png')
    plt.close()
    
    return {
        'importances': importances,
        'indices': indices,
        'feature_names': feature_names
    }

def cross_validate_thresholds(model, X_val, y_val):
    """Perform cross-validation to find optimal thresholds for different metrics"""
    from sklearn.model_selection import KFold
    
    # Create KFold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Metrics to track
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'f_beta': [],
        'threshold_precision': [],
        'threshold_recall': [],
        'threshold_f1': [],
        'threshold_f_beta': []
    }
    
    # Cross-validate
    fold = 1
    for train_idx, test_idx in kf.split(X_val):
        X_fold_train, X_fold_test = X_val[train_idx], X_val[test_idx]
        y_fold_train, y_fold_test = y_val[train_idx], y_val[test_idx]
        
        # Train on validation fold (optional if model is already trained)
        # model.fit(X_fold_train, y_fold_train)
        
        # Get predictions
        y_proba = model.predict_proba(X_fold_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_fold_test, y_proba)
        
        # Calculate F1 and F-beta scores
        f1_scores = []
        f_beta_scores = []  # beta=0.5 for precision focus
        beta = 0.5
        
        for p, r in zip(precision, recall):
            if p + r == 0:
                f1_scores.append(0)
                f_beta_scores.append(0)
            else:
                f1 = 2 * (p * r) / (p + r)
                f_beta = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)
                f1_scores.append(f1)
                f_beta_scores.append(f_beta)
        
        # Find best thresholds
        if len(thresholds) > 0:
            # For precision (target at least 0.9)
            high_prec_indices = np.where(precision[:-1] >= 0.9)[0]
            if len(high_prec_indices) > 0:
                best_prec_idx = high_prec_indices[np.argmax(recall[high_prec_indices])]
                threshold_prec = thresholds[best_prec_idx]
                metrics['threshold_precision'].append(threshold_prec)
                
                # Metrics at this threshold
                y_pred = (y_proba >= threshold_prec).astype(int)
                metrics['precision'].append(precision_score(y_fold_test, y_pred, zero_division=0))
            
            # For recall (target at least 0.7)
            high_recall_indices = np.where(recall[:-1] >= 0.7)[0]
            if len(high_recall_indices) > 0:
                best_recall_idx = high_recall_indices[np.argmax(precision[high_recall_indices])]
                threshold_recall = thresholds[best_recall_idx]
                metrics['threshold_recall'].append(threshold_recall)
                
                # Metrics at this threshold
                y_pred = (y_proba >= threshold_recall).astype(int)
                metrics['recall'].append(recall_score(y_fold_test, y_pred, zero_division=0))
            
            # For F1
            best_f1_idx = np.argmax(f1_scores[:-1])
            threshold_f1 = thresholds[best_f1_idx]
            metrics['threshold_f1'].append(threshold_f1)
            
            # Metrics at this threshold
            y_pred = (y_proba >= threshold_f1).astype(int)
            metrics['f1'].append(f1_score(y_fold_test, y_pred, zero_division=0))
            
            # For F-beta
            best_f_beta_idx = np.argmax(f_beta_scores[:-1])
            threshold_f_beta = thresholds[best_f_beta_idx]
            metrics['threshold_f_beta'].append(threshold_f_beta)
            
            # Metrics at this threshold
            y_pred = (y_proba >= threshold_f_beta).astype(int)
            
            # Calculate F-beta manually
            p = precision_score(y_fold_test, y_pred, zero_division=0)
            r = recall_score(y_fold_test, y_pred, zero_division=0)
            if p + r > 0:
                f_beta = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)
            else:
                f_beta = 0
                
            metrics['f_beta'].append(f_beta)
        
        fold += 1
    
    # Calculate average thresholds
    avg_threshold_precision = np.mean(metrics['threshold_precision']) if metrics['threshold_precision'] else None
    avg_threshold_recall = np.mean(metrics['threshold_recall']) if metrics['threshold_recall'] else None
    avg_threshold_f1 = np.mean(metrics['threshold_f1']) if metrics['threshold_f1'] else None
    avg_threshold_f_beta = np.mean(metrics['threshold_f_beta']) if metrics['threshold_f_beta'] else None
    
    print("Cross-validation threshold results:")
    if avg_threshold_precision:
        print(f"Precision-optimal threshold: {avg_threshold_precision:.4f} (avg precision: {np.mean(metrics['precision']):.4f})")
    if avg_threshold_recall:
        print(f"Recall-optimal threshold: {avg_threshold_recall:.4f} (avg recall: {np.mean(metrics['recall']):.4f})")
    if avg_threshold_f1:
        print(f"F1-optimal threshold: {avg_threshold_f1:.4f} (avg F1: {np.mean(metrics['f1']):.4f})")
    if avg_threshold_f_beta:
        print(f"F-beta-optimal threshold: {avg_threshold_f_beta:.4f} (avg F-beta: {np.mean(metrics['f_beta']):.4f})")
    
    # Return thresholds with focus on precision (F-beta)
    return avg_threshold_f_beta if avg_threshold_f_beta else 0.5