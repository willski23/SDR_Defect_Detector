import h5py
import numpy as np
import matplotlib.pyplot as plt

def analyze_dataset(file_path):
    """Analyze dataset quality and balance"""
    with h5py.File(file_path, 'r') as f:
        # Check dataset structure
        print("Dataset structure:", list(f.keys()))
        
        images = f['images'][:]
        dead_elements = f['dead_elements'][:]
        
        # Check basic stats
        num_images = len(images)
        print(f"Number of images: {num_images}")
        
        # Check for class imbalance
        defect_counts = [np.sum(dead > 0) for dead in dead_elements]
        total_defects = sum(defect_counts)
        total_elements = sum(dead.size for dead in dead_elements)
        print(f"Total defective elements: {total_defects}")
        print(f"Total elements: {total_elements}")
        print(f"Defect ratio: {total_defects/total_elements:.4f}")
        
        # Check for image quality issues
        zero_var_images = sum(1 for img in images if np.var(img) < 1e-8)
        print(f"Images with near-zero variance: {zero_var_images}")
        
        # Check for constant regions in images
        const_region_count = 0
        for img in images[:min(100, num_images)]:  # Check first 100 images
            # Check for rows with constant values
            row_vars = np.var(img, axis=1)
            const_rows = np.sum(row_vars < 1e-8)
            if const_rows > img.shape[0] * 0.1:  # More than 10% constant rows
                const_region_count += 1
        
        print(f"Images with significant constant regions: {const_region_count} (sampled from 100)")
        
        # Plot distribution of defects per image
        plt.figure(figsize=(10, 6))
        plt.hist(defect_counts, bins=range(max(defect_counts)+2))
        plt.xlabel('Number of Defects')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Defects per Image')
        plt.grid(alpha=0.3)
        plt.savefig('defect_distribution.png')
        
        # Plot average image and variance
        if num_images > 0:
            avg_image = np.mean(images, axis=0)
            var_image = np.var(images, axis=0)
            
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(avg_image, cmap='gray')
            plt.title('Average Image')
            plt.colorbar()
            
            plt.subplot(1, 2, 2)
            plt.imshow(var_image, cmap='viridis')
            plt.title('Variance Across Images')
            plt.colorbar()
            
            plt.savefig('dataset_statistics.png')

if __name__ == "__main__":
    # Update this path to match the new HDF5 file location
    file_path = "C:/Users/wbszy/code_projects/Soundcheck/data/processed/dataset.h5"
    analyze_dataset(file_path)