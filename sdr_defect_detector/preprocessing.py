import cv2
import numpy as np

def normalize_image(image):
    """Normalize image for consistent brightness levels"""
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    normalized = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image) + 1e-8) * 255
    normalized = normalized.astype(np.uint8)
    
    return normalized

def extract_roi(image):
    """Extract region of interest - top portion of reverberation pattern"""
    height, width = image.shape
    
    roi_height = min(int(height * 0.3), 100)
    roi = image[:roi_height, :]
    
    return roi

def create_windows(image, window_size=8, overlap=0.5):
    """Divide image into overlapping windows"""
    height, width = image.shape
    step = int(window_size * (1 - overlap))
    
    windows = []
    positions = []
    
    for col in range(0, width - window_size + 1, step):
        window = image[:, col:col+window_size]
        windows.append(window)
        positions.append((col, col+window_size))
    
    return windows, positions