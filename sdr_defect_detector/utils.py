import pickle
import os
import numpy as np

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

def load_model(filename='defect_detection_model.pkl'):
    """Load saved model"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    threshold = model_data['threshold']
    
    print(f"Loaded model trained on {model_data['timestamp']}")
    print(f"Model performance: F1={model_data['metrics'].get('f1', 'N/A'):.4f}")
    
    return model, threshold, model_data

def load_and_prepare_data(data_dir):
    """Load and prepare dataset"""
    # Modify this function to match your data loading approach
    images = []
    defect_labels = []
    
    # Example for folder-organized dataset
    control_dir = os.path.join(data_dir, 'control')
    for filename in os.listdir(control_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(control_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            defect_labels.append([])  # No defects
    
    # Read defect images and positions
    defect_dir = os.path.join(data_dir, 'defects')
    defect_positions_file = os.path.join(data_dir, 'defect_positions.txt')
    
    defect_positions = {}
    with open(defect_positions_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 1:
                filename = parts[0]
                positions = [int(p) for p in parts[1:]]
                defect_positions[filename] = positions
    
    for filename in os.listdir(defect_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(defect_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            
            if filename in defect_positions:
                defect_labels.append(defect_positions[filename])
            else:
                defect_labels.append([])
    
    return images, defect_labels