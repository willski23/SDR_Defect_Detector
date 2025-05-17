import numpy as np
from scipy.signal import find_peaks
from skimage.feature import graycomatrix, graycoprops

def extract_window_features(window, full_image=None, window_position=None):
    """Extract comprehensive features from a window with improved normalization"""
    features = []
    
    # Ensure window is properly normalized to avoid divide-by-zero
    window_norm = window.copy()
    window_std = np.std(window_norm)
    if window_std == 0 or window_std < 1e-8:
        # Apply small random noise to avoid zero variance
        window_norm = window_norm + np.random.normal(0, 0.01, window_norm.shape)
        window_std = np.std(window_norm)
    
    # Brightness-based features
    brightness_features = extract_brightness_features(window)
    features.extend(brightness_features)
    
    # Pattern-based features
    try:
        pattern_features = extract_pattern_features(window)
        features.extend(pattern_features)
    except Exception as e:
        # Add zeros as placeholders
        features.extend([0] * 14)  # Match the number of pattern features
    
    # Contextual features if position info provided
    if full_image is not None and window_position is not None:
        try:
            contextual_features = extract_contextual_features(window, full_image, window_position)
            features.extend(contextual_features)
        except Exception as e:
            features.extend([0] * 10)  # Match the number of contextual features
    
    # Texture features
    try:
        if window.shape[0] >= 8 and window.shape[1] >= 8:
            texture_features = extract_texture_features(window)
            features.extend(texture_features)
        else:
            # Add placeholder zeros for small windows
            features.extend([0] * 6)  # Match the number of texture features
    except Exception as e:
        features.extend([0] * 6)  # Match the number of texture features
    
    # Feature validation - replace any NaN or inf with zeros
    features = np.array(features)
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 0
    
    return features.tolist()

def extract_brightness_features(window):
    """Extract brightness-based features with improved robustness"""
    features = []
    
    flat_window = window.flatten()
    
    if len(flat_window) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]  # Return zeros if window is empty
        
    sorted_pixels = np.sort(flat_window)
    
    top_n = 10
    if len(sorted_pixels) >= top_n:
        top_pixels = sorted_pixels[-top_n:]
        features.append(np.mean(top_pixels))
    else:
        features.append(np.mean(sorted_pixels) if len(sorted_pixels) > 0 else 0)
    
    # Statistical features with safety checks
    mean_val = np.mean(window) if window.size > 0 else 0
    std_val = np.std(window) if window.size > 0 else 0
    features.extend([
        mean_val,
        std_val,
        np.median(window) if window.size > 0 else 0,
        np.percentile(window, 75) if window.size > 0 else 0,
        np.percentile(window, 90) if window.size > 0 else 0
    ])
    
    # Dynamic range with safety check
    if len(flat_window) > 0:
        p95 = np.percentile(flat_window, 95)
        p5 = np.percentile(flat_window, 5)
        features.append(p95 - p5)
    else:
        features.append(0)
    
    # Add coefficient of variation with safety check
    if mean_val > 0 and std_val > 0:
        features.append(std_val / (mean_val + 1e-8))
    else:
        features.append(0)
    
    # Add skewness with safety check
    if len(flat_window) > 0 and std_val > 0:
        mean = np.mean(flat_window)
        skewness = np.mean(((flat_window - mean) / (std_val + 1e-8)) ** 3)
        features.append(skewness)
    else:
        features.append(0)
    
    return features

def extract_pattern_features(window):
    """Extract features related to reverberation pattern with enhanced robustness"""
    features = []
    height, width = window.shape
    
    # Vertical profile
    vertical_profile = np.mean(window, axis=1)
    
    # Safety check for empty profile
    if len(vertical_profile) < 3:
        return [0] * 14  # Return zeros if profile is too small
    
    # Set safe default values for statistics
    mean_profile = np.mean(vertical_profile)
    std_profile = np.std(vertical_profile)
    
    # Find peaks with safe parameters
    try:
        peaks, properties = find_peaks(
            vertical_profile, 
            height=mean_profile + 0.2 * std_profile,
            distance=2,  # Minimum distance between peaks
            prominence=std_profile * 0.3  # Lower prominence threshold
        )
    except Exception:
        peaks = []
    
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
    
    # Improved horizontal uniformity with safety checks
    row_variations = np.std(window, axis=1)
    features.append(np.mean(row_variations) if row_variations.size > 0 else 0)
    
    # Add normalized row variations with safety check
    if np.mean(window) > 0:
        norm_row_variations = np.std(window, axis=1) / (np.mean(window) + 1e-8)
        features.append(np.mean(norm_row_variations) if norm_row_variations.size > 0 else 0)
    else:
        features.append(0)
    
    # Add local consistency feature with safety check
    local_consistency = []
    for i in range(1, height-1):
        # Use safe correlation function
        consistency = safe_correlation(window[i-1], window[i])
        local_consistency.append(consistency)
    
    features.append(np.mean(local_consistency) if local_consistency else 0)
    
    # Add peak regularity with safety check
    if len(peaks) > 2:
        peak_distances = np.diff(peaks)
        features.append(np.std(peak_distances) / (np.mean(peak_distances) + 1e-8))
    else:
        features.append(0)
    
    return features

def extract_contextual_features(window, full_image, window_position):
    """Extract enhanced contextual features with robust error handling"""
    features = []
    height, width = window.shape
    start_col, end_col = window_position
    
    # Safety checks
    if start_col >= full_image.shape[1] or end_col > full_image.shape[1] or height > full_image.shape[0]:
        return [0] * 10  # Return zeros for invalid dimensions
    
    # Get regions to left and right with safe boundaries
    left_width = min(width*2, start_col)
    right_width = min(width*2, full_image.shape[1] - end_col)
    
    left_region = None
    if left_width > 0:
        left_region = full_image[:min(height, full_image.shape[0]), max(0, start_col-left_width):start_col]
    
    right_region = None
    if right_width > 0:
        right_region = full_image[:min(height, full_image.shape[0]), end_col:min(full_image.shape[1], end_col+right_width)]
    
    # Brightness ratio features with safety checks
    window_brightness = np.mean(window) if window.size > 0 else 0
    
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
    
    # Edge continuity features with proper boundary checks
    if left_region is not None and np.size(left_region) > 0 and left_region.shape[1] > 0:
        edge_width = min(3, left_region.shape[1])
        left_edge = left_region[:, -edge_width:]
        window_left_edge = window[:, :min(3, window.shape[1])]
        
        if left_edge.size > 0 and window_left_edge.size > 0:
            edge_diff = np.abs(np.mean(left_edge) - np.mean(window_left_edge))
            features.append(edge_diff)
        else:
            features.append(0)
    else:
        features.append(0)
    
    if right_region is not None and np.size(right_region) > 0 and right_region.shape[1] > 0:
        edge_width = min(3, right_region.shape[1])
        right_edge = right_region[:, :edge_width]
        window_right_edge = window[:, -min(3, window.shape[1]):]
        
        if right_edge.size > 0 and window_right_edge.size > 0:
            edge_diff = np.abs(np.mean(right_edge) - np.mean(window_right_edge))
            features.append(edge_diff)
        else:
            features.append(0)
    else:
        features.append(0)
    
    # Standard deviation ratio with safety checks
    window_std = np.std(window) if window.size > 0 else 0
    
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
    
    # Pattern consistency with neighbors using safe correlation
    window_pattern = np.mean(window, axis=1)  # Vertical pattern
    
    if left_region is not None and np.size(left_region) > 0:
        left_pattern = np.mean(left_region, axis=1)
        if len(window_pattern) == len(left_pattern) and len(window_pattern) > 0:
            corr = safe_correlation(window_pattern, left_pattern)
            features.append(corr)
        else:
            features.append(0)
    else:
        features.append(0)
    
    if right_region is not None and np.size(right_region) > 0:
        right_pattern = np.mean(right_region, axis=1)
        if len(window_pattern) == len(right_pattern) and len(window_pattern) > 0:
            corr = safe_correlation(window_pattern, right_pattern)
            features.append(corr)
        else:
            features.append(0)
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
    """Extract texture features using GLCM with improved error handling"""
    features = []
    
    min_size = 8
    if window.shape[0] < min_size or window.shape[1] < min_size:
        return [0, 0, 0, 0, 0, 0]
    
    # Scale window values to appropriate range for GLCM
    window_min = np.min(window)
    window_max = np.max(window)
    
    if window_max == window_min:
        return [0, 0, 0, 0, 0, 0]  # Return zeros for constant window
    
    levels = 32
    window_scaled = np.clip(((window - window_min) / (window_max - window_min) * (levels-1)).astype(np.uint8), 0, levels-1)
    
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
        
    except Exception as e:
        features.extend([0, 0, 0, 0, 0, 0])
    
    return features

def safe_correlation(a, b):
    """Calculate correlation with proper handling of edge cases"""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    
    # Handle constant arrays
    std_a = np.std(a)
    std_b = np.std(b)
    
    if std_a < 1e-8 or std_b < 1e-8:
        if np.allclose(a, b, rtol=1e-5, atol=1e-8):
            return 1.0  # Perfect correlation for identical constant arrays
        else:
            return 0.0  # No correlation between different constants
    
    # Calculate correlation normally
    try:
        corr = np.corrcoef(a, b)[0, 1]
        
        # Handle NaN result
        if np.isnan(corr):
            return 0.0
            
        return corr
    except:
        return 0.0