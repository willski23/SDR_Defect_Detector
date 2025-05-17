import numpy as np

def simple_element_features(image, element_position):
    """Extract simple but effective element-specific features"""
    if image is None or element_position < 0 or element_position >= image.shape[1]:
        return np.zeros(10)
        
    # Extract element column
    element_col = image[:, element_position]
    
    # Basic statistical features
    features = []
    features.append(np.mean(element_col))
    features.append(np.std(element_col))
    features.append(np.median(element_col))
    features.append(np.max(element_col))
    features.append(np.min(element_col))
    
    # Comparison with neighbors
    if element_position > 0:
        left_col = image[:, element_position-1]
        features.append(np.mean(np.abs(element_col - left_col)))
    else:
        features.append(0)
        
    if element_position < image.shape[1]-1:
        right_col = image[:, element_position+1]
        features.append(np.mean(np.abs(element_col - right_col)))
    else:
        features.append(0)
    
    # Normalize position (0 = edge, 1 = center)
    norm_pos = 2 * min(element_position, image.shape[1]-1-element_position) / image.shape[1]
    features.append(norm_pos)
    
    # Add top vs bottom comparison
    if len(element_col) >= 4:
        top_half = element_col[:len(element_col)//2]
        bottom_half = element_col[len(element_col)//2:]
        features.append(np.mean(top_half) / (np.mean(bottom_half) + 1e-8))
    else:
        features.append(1.0)
    
    # Special contrast feature - difference to average
    col_avg = np.mean(image, axis=1)
    if len(col_avg) == len(element_col):
        features.append(np.mean(np.abs(element_col - col_avg)))
    else:
        features.append(0)
    
    # Ensure we have exactly 10 features and handle invalid values
    features = np.array(features[:10])  # Limit to first 10 if we have more
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 0
    
    if len(features) < 10:
        features = np.pad(features, (0, 10 - len(features)))
    
    return features