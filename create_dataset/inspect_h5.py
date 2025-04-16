import h5py
import numpy as np
import matplotlib.pyplot as plt

def inspect_h5(file_path):
    """Inspect the contents of an HDF5 file"""
    with h5py.File(file_path, 'r') as f:
        # List all groups and datasets
        print("HDF5 File Structure:")
        print("====================")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                data_type = obj.dtype
                shape = obj.shape
                print(f"Dataset: {name}")
                print(f"  Shape: {shape}")
                print(f"  Type: {data_type}")
                
                # For numeric datasets, print some statistics
                if np.issubdtype(data_type, np.number) and len(obj) > 0:
                    if obj.size < 1000:  # Only load small datasets fully
                        data = obj[:]
                        print(f"  Min: {np.min(data)}")
                        print(f"  Max: {np.max(data)}")
                        print(f"  Mean: {np.mean(data)}")
                    else:
                        # For larger datasets, read a sample
                        sample = obj[0:min(10, obj.shape[0])]
                        print(f"  Sample min: {np.min(sample)}")
                        print(f"  Sample max: {np.max(sample)}")
                        print(f"  Sample mean: {np.mean(sample)}")
                
                # For string datasets, print a few examples
                if data_type.kind == 'S' or data_type.kind == 'O':
                    if len(obj) > 0:
                        print(f"  First few values: {obj[0:min(5, len(obj))]}")
            else:
                print(f"Group: {name}")
        
        # Recursively visit all objects in the file
        f.visititems(print_structure)
        
        # Visualize some samples if datasets exist
        if 'images' in f and 'dead_elements' in f:
            images = f['images']
            dead_elements = f['dead_elements']
            filenames = f['filenames'] if 'filenames' in f else None
            
            if len(images) > 0:
                # Plot a few samples
                num_samples = min(3, len(images))
                for i in range(num_samples):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Plot image
                    im = ax1.imshow(images[i], cmap='gray')
                    title = f"Sample {i+1} - Image"
                    if filenames is not None:
                        title += f" - {filenames[i]}"
                    ax1.set_title(title)
                    plt.colorbar(im, ax=ax1)
                    
                    # Plot dead elements
                    ax2.bar(range(len(dead_elements[i])), dead_elements[i])
                    ax2.set_title(f"Sample {i+1} - Dead Elements")
                    ax2.set_xlabel("Transducer Index")
                    ax2.set_ylabel("Status (1 = disabled)")
                    ax2.set_ylim(0, 1.1)
                    
                    plt.tight_layout()
                    plt.show()

if __name__ == "__main__":
    # Update this path to match the new HDF5 file location
    file_path = "C:/Users/wbszy/code_projects/Soundcheck/data/processed/dataset.h5"
    inspect_h5(file_path)