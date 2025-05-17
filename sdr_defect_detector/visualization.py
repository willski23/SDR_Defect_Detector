import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def visualize_defects(image, defect_positions):
    """Create visualization of detected defects with improved color coding"""
    vis_image = image.copy()
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Mark defect positions
    height = vis_image.shape[0]
    for pos, conf in defect_positions:
        # Color based on confidence (green->yellow->red)
        # Higher confidence = more red (better visibility)
        color_val = min(255, int(255 * max(conf, 0.5)))
        color = (0, 255-color_val, color_val)  # BGR format
        
        # Draw vertical line at defect position
        cv2.line(vis_image, (pos, 0), (pos, height//3), color, 2)
        
        # Add confidence text with improved positioning
        conf_text = f"{conf:.2f}"
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = max(0, min(pos - text_size[0]//2, vis_image.shape[1] - text_size[0]))
        cv2.putText(vis_image, conf_text, (text_x, height//3+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis_image

def visualize_feature_importance(model, feature_names=None):
    """Visualize feature importance from model with improved robustness"""
    # For calibrated models, access the base estimator
    if hasattr(model, 'base_estimator'):
        # For calibrated models, get the first calibrated estimator
        if hasattr(model, 'calibrated_classifiers_'):
            base_model = model.calibrated_classifiers_[0].base_estimator
        else:
            base_model = model.base_estimator
    else:
        base_model = model
    
    # Make sure the model has feature_importances_ attribute
    if not hasattr(base_model, 'feature_importances_'):
        # Create dummy plot for models without feature importances
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance Not Available for This Model Type')
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Feature')
        return plt
    
    # Now get feature importances from the base model
    importances = base_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # If feature names are not provided, use indices
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    # Plot top 20 features (or fewer if less are available)
    n_features = min(20, len(importances))
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.barh(range(n_features), importances[indices[:n_features]], align='center')
    plt.yticks(range(n_features), [feature_names[i] for i in indices[:n_features]])
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    return plt

def visualize_confusion_matrix(y_true, y_pred, classes=['Normal', 'Defect']):
    """Visualize confusion matrix with improved styling"""
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages for annotation
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm, dtype=str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax)
    
    # Labels, title and ticks
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Fix for matplotlib/seaborn bug
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    return plt

def visualize_confidence_distribution(defect_positions, true_defects=None):
    """Visualize confidence distribution of detected defects"""
    # Extract confidences
    confidences = [conf for _, conf in defect_positions]
    
    if not confidences:
        # Create empty plot if no defects detected
        plt.figure(figsize=(10, 6))
        plt.title('No Defects Detected - Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        return plt
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # If ground truth available, separate true and false positives
    if true_defects is not None:
        # Create lists for true and false positives
        true_pos_conf = []
        false_pos_conf = []
        
        # Categorize detections
        for pos, conf in defect_positions:
            if pos in true_defects:
                true_pos_conf.append(conf)
            else:
                false_pos_conf.append(conf)
        
        # Plot histograms
        bins = np.linspace(0, 1, 20)
        if true_pos_conf:
            plt.hist(true_pos_conf, bins=bins, alpha=0.7, 
                    label='True Positives', color='green')
        if false_pos_conf:
            plt.hist(false_pos_conf, bins=bins, alpha=0.7, 
                    label='False Positives', color='red')
        
        plt.legend()
        plt.title('Confidence Distribution - True vs False Positives')
    else:
        # Plot single histogram for all detections
        plt.hist(confidences, bins=20, alpha=0.7, color='blue')
        plt.title('Confidence Distribution of Detected Defects')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    
    return plt

def visualize_precision_recall_curve(precision, recall, thresholds, threshold=None):
    """Visualize precision-recall curve with optional selected threshold"""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, marker='.', linewidth=1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    # Mark selected threshold if provided
    if threshold is not None:
        # Find closest threshold
        selected_idx = np.abs(thresholds - threshold).argmin()
        if selected_idx < len(precision) and selected_idx < len(recall):
            plt.plot(recall[selected_idx], precision[selected_idx], 'ro', markersize=8, 
                     label=f'Selected Threshold: {threshold:.2f}')
    
    # Add gridlines and labels
    plt.grid(alpha=0.3)
    plt.title('Precision-Recall Curve')
    if threshold is not None:
        plt.legend()
    
    # Set axis limits
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    return plt