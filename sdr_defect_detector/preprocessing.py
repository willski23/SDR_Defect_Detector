import cv2
import numpy as np

def normalize_image(image):
    """Normalize image for consistent brightness levels with robust handling"""
    if image is None or image.size == 0:
        # Return empty image for invalid input
        return np.zeros((1, 1), dtype=np.uint8)
    
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    # Handle edge case of constant image
    img_min = np.min(gray_image)
    img_max = np.max(gray_image)
    
    if img_max <= img_min:
        # Avoid division by zero for constant images
        return np.zeros_like(gray_image, dtype=np.uint8)
    
    normalized = (gray_image - img_min) / (img_max - img_min + 1e-8) * 255
    normalized = normalized.astype(np.uint8)
    
    return normalized

def extract_roi(image):
    """Extract region of interest with robust error handling"""
    if image is None or image.size == 0:
        # Return empty image for invalid input
        return np.zeros((1, 1), dtype=np.uint8)
    
    height, width = image.shape
    
    # Ensure valid ROI dimensions
    roi_height = max(1, min(int(height * 0.3), 100))
    
    # Extract ROI with bounds checking
    if roi_height > height:
        roi_height = height
    
    roi = image[:roi_height, :]
    
    return roi

def create_windows(image, window_size=8, overlap=0.5):
    """Divide image into overlapping windows with improved boundaries handling"""
    if image is None or image.size == 0 or window_size <= 0:
        # Return empty data for invalid input
        return [], []
    
    height, width = image.shape
    
    # Ensure valid step size
    step = max(1, int(window_size * (1 - min(overlap, 0.95))))
    
    windows = []
    positions = []
    
    for col in range(0, max(0, width - window_size) + 1, step):
        # Ensure the window doesn't go beyond the image boundary
        end_col = min(col + window_size, width)
        window = image[:, col:end_col]
        
        # Only add valid windows
        if window.shape[1] > 0:
            windows.append(window)
            positions.append((col, end_col))
    
    # Add a final window at the right edge if needed and not already covered
    if width > 0 and (width - window_size) % step != 0 and width > window_size:
        end_col = width
        start_col = end_col - window_size
        if start_col >= 0 and (start_col, end_col) not in positions:
            window = image[:, start_col:end_col]
            windows.append(window)
            positions.append((start_col, end_col))
    
    return windows, positions