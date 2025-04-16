import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import os
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

class UltrasoundDefectDataset(Dataset):
    """
    Custom Dataset for ultrasound defect detection
    Loads MATLAB files and converts them directly to PyTorch tensors
    """
    def __init__(self, data_dir, output_dir, transform=None, use_saved=False):
        """
        Args:
            data_dir (str): Directory with all the .mat files
            output_dir (str): Directory to save the processed data
            transform (callable, optional): Optional transform to be applied on a sample
            use_saved (bool): Whether to use saved processed data
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.transform = transform
        
        # Define the path for the processed dataset file
        self.processed_file_path = os.path.join(self.output_dir, 'processed', 'dataset.h5')
        os.makedirs(os.path.dirname(self.processed_file_path), exist_ok=True)
        
        if use_saved and os.path.exists(self.processed_file_path):
            self.load_processed_data()
        else:
            self.process_data()
            
    def process_data(self):
        """Process all .mat files and save to HDF5 format"""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.processed_file_path), exist_ok=True)
        
        # Get list of all .mat files
        file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.mat')]
        
        print(f"Found {len(file_list)} .mat files")
        
        # Process and save all files
        with h5py.File(self.processed_file_path, 'w') as f:
            # Create datasets
            images_dataset = f.create_dataset('images', (len(file_list), 118, 128), dtype='float32')
            dead_elements_dataset = f.create_dataset('dead_elements', (len(file_list), 128), dtype='uint8')
            filenames_dataset = f.create_dataset('filenames', (len(file_list),), dtype=h5py.special_dtype(vlen=str))
            
            print("Processing .mat files...")
            for i, filename in enumerate(tqdm(file_list)):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    # Load .mat file
                    mat_data = sio.loadmat(file_path)
                    
                    # Extract image data and dead elements data using variable names from your files
                    image_data = mat_data['imgData']  # 118x128 array
                    dead_elements = mat_data['deadElements'].flatten().astype(np.uint8)  # 1x128 array
                    
                    # Normalize image data to [0, 1] range
                    min_val = np.min(image_data)
                    max_val = np.max(image_data)
                    if max_val > min_val:  # Avoid division by zero
                        image_data = (image_data - min_val) / (max_val - min_val)
                    
                    # Store in HDF5 file
                    images_dataset[i] = image_data
                    dead_elements_dataset[i] = dead_elements
                    filenames_dataset[i] = filename
                    
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
        
        self.load_processed_data()
    
    def load_processed_data(self):
        """Load the processed data directly into PyTorch tensors"""
        print(f"Loading processed data from {self.processed_file_path}...")
        
        with h5py.File(self.processed_file_path, 'r') as f:
            # Load all data into memory as PyTorch tensors
            self.images = torch.tensor(f['images'][:], dtype=torch.float32)
            self.dead_elements = torch.tensor(f['dead_elements'][:], dtype=torch.uint8) 
            self.filenames = [filename for filename in f['filenames'][:]]
        
        self.length = len(self.images)
        print(f"Loaded dataset with {self.length} samples")
        
        # Count number of files with dead elements
        files_with_defects = torch.sum(torch.sum(self.dead_elements, dim=1) > 0).item()
        print(f"Files with one or more dead elements: {files_with_defects}/{self.length}")
    
    def __len__(self):
        """Return the total number of samples"""
        return self.length
    
    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        image = self.images[idx]
        dead_elements = self.dead_elements[idx]
        
        # Create sample dictionary
        sample = {
            'image': image,
            'dead_elements': dead_elements,
            'filename': self.filenames[idx] if hasattr(self, 'filenames') else ''
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def visualize_samples(dataset, num_samples=4):
    """Visualize samples from the dataset"""
    if len(dataset) == 0:
        print("Dataset is empty, nothing to visualize")
        return
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Visualize each sample
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        image = sample['image']
        dead_elements = sample['dead_elements']
        filename = sample['filename']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot the image
        im = ax1.imshow(image, cmap='gray')
        ax1.set_title(f"Ultrasound Image - {filename}")
        plt.colorbar(im, ax=ax1)
        
        # Plot the dead elements
        ax2.bar(range(len(dead_elements)), dead_elements)
        ax2.set_title("Dead Elements (1 = disabled)")
        ax2.set_xlabel("Transducer Index")
        ax2.set_ylabel("Status (1 = disabled)")
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Sample {i+1}/{len(indices)}: {filename}")
        print(f"  Image range: [{torch.min(image).item():.4f}, {torch.max(image).item():.4f}]")
        print(f"  Number of disabled transducers: {torch.sum(dead_elements).item()}")

# Example usage
if __name__ == "__main__":
    # Set the data directories
    data_dir = "C:/Users/wbszy/code_projects/Soundcheck/data/matfiles"
    output_dir = "C:/Users/wbszy/code_projects/Soundcheck/data"
    
    # Create dataset
    dataset = UltrasoundDefectDataset(
        data_dir=data_dir,
        output_dir=output_dir,
        use_saved=False  # Process the files again
    )
    
    # Visualize some samples
    # visualize_samples(dataset, num_samples=4)