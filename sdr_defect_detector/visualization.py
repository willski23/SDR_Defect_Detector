import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_defects(image, defect_positions):
    """Create visualization of detected defects"""
    vis_image = image.copy()
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Mark defect positions
    height = vis_image.shape[0]
    for pos, conf in defect_positions:
        # Color based on confidence (green->yellow->red)
        color_val = int(255 * conf)
        color = (0, 255-color_val, color_val)  # BGR format
        
        # Draw vertical line at defect position
        cv2.line(vis_image, (pos, 0), (pos, height//4), color, 2)
        
        # Optional: Add confidence text
        if conf > 0.8:  # Only show high confidence
            cv2.putText(vis_image, f"{conf:.2f}", (pos-10, height//4+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis_image

def visualize_feature_importance(model, feature_names=None):
    """Visualize feature importance from model"""
    importances = model.feature_importances_
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
    """Visualize confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    return plt