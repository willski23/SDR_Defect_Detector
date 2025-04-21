import numpy as np
from scipy.signal import find_peaks
from skimage.feature import graycomatrix, graycoprops

def extract_window_features(window, full_image=None, window_position=None):
    """Extract comprehensive features from a window"""
    features = []
    
    # Brightness-based features
    brightness_features = extract_brightness_features(window)
    features.extend(brightness_features)
    
    # Pattern-based features
    pattern_features = extract_pattern_features(window)
    features.extend(pattern_features)
    
    # Contextual features if position info provided
    if full_image is not None and window_position is not None:
        contextual_features = extract_contextual_features(window, full_image, window_position)
        features.extend(contextual_features)
    
    # Texture features
    texture_features = extract_texture_features(window)
    features.extend(texture_features)
    
    return features

def extract_brightness_features(window):
    """Extract brightness-based features"""
    features = []
    
    flat_window = window.flatten()
    sorted_pixels = np.sort(flat_window)
    
    top_n = 10
    if len(sorted_pixels) >= top_n:
        top_pixels = sorted_pixels[-top_n:]
        features.append(np.mean(top_pixels))
    else:
        features.append(0)
    
    # Statistical features
    features.extend([
        np.mean(window),
        np.std(window),
        np.median(window),
        np.percentile(window, 75),
        np.percentile(window, 90)
    ])
    
    # Dynamic range
    if len(flat_window) > 0:
        p95 = np.percentile(flat_window, 95)
        p5 = np.percentile(flat_window, 5)
        features.append(p95 - p5)
    else:
        features.append(0)
    
    # Add coefficient of variation
    if np.mean(window) > 0:
        features.append(np.std(window) / (np.mean(window) + 1e-8))
    else:
        features.append(0)
    
    # Add skewness
    if len(flat_window) > 0:
        mean = np.mean(flat_window)
        std = np.std(flat_window)
        if std > 0:
            skewness = np.mean(((flat_window - mean) / std) ** 3)
            features.append(skewness)
        else:
            features.append(0)
    else:
        features.append(0)
    
    return features

def extract_pattern_features(window):
    """Extract features related to reverberation pattern with improved precision"""
    features = []
    height, width = window.shape
    
    # Vertical profile
    vertical_profile = np.mean(window, axis=1)
    
    # Find peaks (reverberation bands) with improved parameters
    peaks, properties = find_peaks(
        vertical_profile, 
        height=np.mean(vertical_profile) + 0.5 * np.std(vertical_profile),
        distance=2,  # Minimum distance between peaks
        prominence=np.std(vertical_profile) * 0.5  # Added prominence threshold
    )
    
    features.append(len(peaks))
    
    if len(peaks) > 1:
        peak_distances = np.diff(peaks)
        features.extend([
            np.mean(peak_distances),
            np.std(peak_distances),
            np.min(peak_distances),
            np.max(peak_distances)
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    if len(peaks) > 0:
        peak_values = vertical_profile[peaks]
        features.extend([
            np.mean(peak_values),
            np.std(peak_values),
            np.min(peak_values),
            np.max(peak_values)
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    # Improved horizontal uniformity
    row_variations = np.std(window, axis=1)
    features.append(np.mean(row_variations))
    
    # Add normalized row variations
    if np.mean(window) > 0:
        norm_row_variations = np.std(window, axis=1) / (np.mean(window) + 1e-8)
        features.append(np.mean(norm_row_variations))
    else:
        features.append(0)
    
    # Add local consistency feature
    local_consistency = []
    for i in range(1, height-1):
        consistency = np.corrcoef(window[i-1], window[i])[0,1]
        if np.isnan(consistency):
            consistency = 0
        local_consistency.append(consistency)
    
    features.append(np.mean(local_consistency) if local_consistency else 0)
    
    # Add peak regularity (coefficient of variation of peak distances)
    if len(peaks) > 2:
        peak_distances = np.diff(peaks)
        features.append(np.std(peak_distances) / (np.mean(peak_distances) + 1e-8))
    else:
        features.append(0)
    
    return features

def extract_contextual_features(window, full_image, window_position):
    """Extract enhanced contextual features comparing window to surroundings"""
    features = []
    height, width = window.shape
    start_col, end_col = window_position
    
    # Get regions to left and right (expanded range)
    left_width = min(width*2, start_col)
    right_width = min(width*2, full_image.shape[1] - end_col)
    
    left_region = None
    if left_width > 0:
        left_region = full_image[:height, start_col-left_width:start_col]
    
    right_region = None
    if right_width > 0:
        right_region = full_image[:height, end_col:end_col+right_width]
    
    # Brightness ratio features
    window_brightness = np.mean(window)
    
    if left_region is not None and np.size(left_region) > 0:
        left_brightness = np.mean(left_region)
        features.append(window_brightness / (left_brightness + 1e-8))
    else:
        features.append(1.0)
    
    if right_region is not None and np.size(right_region) > 0:
        right_brightness = np.mean(right_region)
        features.append(window_brightness / (right_brightness + 1e-8))
    else:
        features.append(1.0)
    
    # Edge continuity features
    if left_region is not None and np.size(left_region) > 0:
        left_edge = left_region[:, -min(3, left_region.shape[1]):]
        window_left_edge = window[:, :min(3, window.shape[1])]
        
        edge_diff = np.abs(np.mean(left_edge) - np.mean(window_left_edge))
        features.append(edge_diff)
    else:
        features.append(0)
    
    if right_region is not None and np.size(right_region) > 0:
        right_edge = right_region[:, :min(3, right_region.shape[1])]
        window_right_edge = window[:, -min(3, window.shape[1]):]
        
        edge_diff = np.abs(np.mean(right_edge) - np.mean(window_right_edge))
        features.append(edge_diff)
    else:
        features.append(0)
    
    #standard deviation ratio
    window_std = np.std(window)
    
    if left_region is not None and np.size(left_region) > 0:
        left_std = np.std(left_region)
        features.append(window_std / (left_std + 1e-8))
    else:
        features.append(1.0)
    
    if right_region is not None and np.size(right_region) > 0:
        right_std = np.std(right_region)
        features.append(window_std / (right_std + 1e-8))
    else:
        features.append(1.0)
    
    # Pattern consistency with neighbors
    window_pattern = np.mean(window, axis=1)  # Vertical pattern
    
    if left_region is not None and np.size(left_region) > 0:
        left_pattern = np.mean(left_region, axis=1)
        corr = np.corrcoef(window_pattern, left_pattern)[0,1]
        features.append(0 if np.isnan(corr) else corr)
    else:
        features.append(0)
    
    if right_region is not None and np.size(right_region) > 0:
        right_pattern = np.mean(right_region, axis=1)
        corr = np.corrcoef(window_pattern, right_pattern)[0,1]
        features.append(0 if np.isnan(corr) else corr)
    else:
        features.append(0)
    
    # Add position-based feature (distance from edge)
    image_width = full_image.shape[1]
    center_pos = (start_col + end_col) / 2
    # Normalize distance from edge (closer to 1 means closer to edge)
    edge_distance = min(center_pos, image_width - center_pos) / (image_width / 2)
    features.append(edge_distance)
    
    return features

def extract_texture_features(window):
    """Extract texture features using GLCM"""
    features = []
    
    min_size = 8
    if window.shape[0] < min_size or window.shape[1] < min_size:
        return [0, 0, 0, 0, 0]
    
    levels = 32
    window_scaled = (window / 256 * levels).astype(np.uint8)
    
    try:
        glcm = graycomatrix(
            window_scaled, 
            distances=[1], 
            angles=[0, np.pi/2],
            levels=levels,
            symmetric=True,
            normed=True
        )
        
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
        
        # Add ASM (Angular Second Moment)
        asm = graycoprops(glcm, 'ASM').mean()
        features.append(asm)
        
    except:
        features.extend([0, 0, 0, 0, 0, 0])
    
    return features