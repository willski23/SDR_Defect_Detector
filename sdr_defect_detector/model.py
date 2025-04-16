import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

def train_defect_detection_model(X_train, y_train):
    """Train and optimize defect detection model"""
    # Check for class imbalance
    from collections import Counter
    class_counts = Counter(y_train)
    print(f"Class distribution: {class_counts}")
    
    # Handle class imbalance if necessary
    if class_counts[1] / sum(class_counts.values()) < 0.3:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {Counter(y_train)}")
    
    # Basic model with reasonable defaults
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    # Hyperparameter optimization
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    
    # Train the model
    print("Training model...")
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