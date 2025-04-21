# Ultrasound Transducer Defect Detection

This repository contains precision-optimized scripts for detecting defective elements in ultrasound transducer images using an HDF5 dataset structure.

## Improved Features

The code has been enhanced to improve precision (reduce false positives) while maintaining good recall:

1. **Enhanced Feature Extraction:**
   - Improved pattern recognition with better peak detection
   - Added normalized feature variations
   - Enhanced contextual comparison with neighboring elements
   - Improved texture analysis

2. **Precision-Focused Training:**
   - Customized class weights to reduce false positives
   - Optimized sampling strategy for imbalanced data
   - Probability calibration for better confidence estimates
   - Custom F1 scorer with precision focus

3. **Smart Thresholding:**
   - Position-aware thresholding (stricter near edges)
   - Feature-specific confidence adjustment
   - Isolated detection filtering

## Requirements

Required Python packages:
```
numpy>=1.20.0
h5py>=3.1.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
scipy>=1.7.0
scikit-image>=0.18.0
matplotlib>=3.4.0
tqdm>=4.60.0
imbalanced-learn>=0.8.0
joblib>=1.0.0
pandas>=1.3.0
seaborn>=0.11.0
```

Install with:
```
pip install -r requirements.txt
```

## Dataset Structure

The system expects an HDF5 file with the following structure:

- `images`: Dataset containing ultrasound transducer images
- `dead_elements`: Dataset containing information about defective elements for each image (non-zero values indicate defects)
- `filenames`: (Optional) Dataset containing filenames for each image

## Usage

### 1. Training a New Model

Train a model with enhanced precision focus:

```bash
python train_model_improved.py --data_file dataset.h5 --output model_precision.pkl --precision_focus 0.8
```

Parameters:
- `--save_vis`: Save visualization images (flag)
- `--precision_mode`: Enable precision mode for stricter detection criteria (flag)

## Results and Output

The batch processing script generates the following outputs:

1. `detection_results.json`: Detailed results for each processed image, including:
   - Detection positions and confidence scores
   - Comparison with ground truth (if available)
   - Performance metrics (precision, recall, F1 score)

2. Summary plots (if ground truth is available):
   - `defect_distribution.png`: Distribution of defects per image
   - `precision_recall.png`: Precision vs. Recall by image
   - `f1_distribution.png`: Distribution of F1 scores
   - `confidence_accuracy.png`: Relationship between confidence scores and detection accuracy

3. Visualization images for each processed image (if `--save_vis` is specified)

## Usage Tips for Improved Precision

To get the best precision-recall balance:

1. **Threshold Adjustment**
   - The model determines an optimal threshold during training, but you can fine-tune it:
   - Increase threshold (e.g., `--threshold 0.7`) to reduce false positives
   - Decrease threshold (e.g., `--threshold 0.5`) if you're missing too many defects

2. **Feature Importance**
   - The model generates a `feature_importance.png` file showing most influential features
   - These features can help understand what signals are driving detections

3. **Confidence Scores**
   - Higher confidence scores (>0.8) are more reliable
   - Examine lower confidence detections (0.5-0.7) more carefully

4. **Batch Processing**
   - Use the `--precision_mode` flag to enable stricter detection criteria
   - Review the performance metrics to find optimal settings

## Implementation Details

The improvement in precision is achieved through multiple strategies:

1. **Enhanced Feature Extraction**
   - `feature_extraction_improved.py`: Adds new features and improves existing ones
   - New features include normalized variations, pattern consistency, and edge distance

2. **Precision-Focused Training**
   - `train_model_improved.py`: Uses class weights and custom scoring function
   - Applies probability calibration to make confidence scores more reliable

3. **Smart Detection**
   - `detect_image_improved.py`: Implements position-aware and feature-specific thresholding
   - Filters isolated detections and applies post-processing

4. **Comprehensive Evaluation**
   - `batch_process_improved.py`: Provides detailed performance metrics and visualizations
   - New confidence-accuracy plot helps identify optimal confidence thresholds

## Troubleshooting

- If you encounter out-of-bounds errors with feature indices, edit the detection script to use the correct feature indices
- For memory issues with large datasets, process in smaller batches using `--start_idx` and `--end_idx`
- If precision is still too low, try increasing the `--precision_focus` value during training (up to 1.0)
- If you're missing many defects, try reducing the threshold or retraining with lower precision focus

## Example Workflow

1. Train a precision-focused model:
```bash
python train_model_improved.py --data_file dataset.h5 --output model_precision.pkl --precision_focus 0.8
```

2. Test on a single image with visualization:
```bash
python detect_image_improved.py --model model_precision.pkl --data_file dataset.h5 --image_index 42
```

3. Process all images with enhanced precision:
```bash
python batch_process_improved.py --model model_precision.pkl --data_file dataset.h5 --output_dir results --save_vis --precision_mode
```

4. Fine-tune threshold based on results:
```bash
python batch_process_improved.py --model model_precision.pkl --data_file dataset.h5 --output_dir results_tuned --threshold 0.65
```data_file`: Path to HDF5 data file (required)
- `--output`: Output model filename (default: 'defect_detection_model.pkl')
- `--window_size`: Window size for feature extraction (default: 8)
- `--overlap`: Window overlap ratio (default: 0.5)
- `--precision_focus`: Focus on precision (0-1, higher values prioritize precision) (default: 0.7)

### 2. Single Image Detection

Detect defects in a single image with improved precision:

```bash
python detect_image_improved.py --model model_precision.pkl --data_file dataset.h5 --image_index 10 --output detected.png
```

Parameters:
- `--model`: Model file path (default: 'defect_detection_model.pkl')
- `--data_file`: HDF5 data file containing images (required)
- `--image_index`: Index of the image to analyze in the dataset (default: 0)
- `--output`: Path to save visualization output (optional)
- `--window_size`: Window size for feature extraction (default: 8)
- `--overlap`: Window overlap ratio (default: 0.5)
- `--threshold`: Custom threshold (overrides saved threshold)
- `--precision_mode`: Enable precision mode (stricter detection criteria)

### 3. Batch Processing

Process multiple images with enhanced precision:

```bash
python batch_process_improved.py --model model_precision.pkl --data_file dataset.h5 --output_dir results --save_vis --precision_mode
```

Parameters:
- `--model`: Model file path (default: 'defect_detection_model.pkl')
- `--data_file`: HDF5 data file containing images (required)
- `--output_dir`: Directory to save output (default: 'output')
- `--window_size`: Window size for feature extraction (default: 8)
- `--overlap`: Window overlap ratio (default: 0.5)
- `--threshold`: Custom threshold (overrides saved threshold)
- `--start_idx`: Start index in dataset (default: 0)
- `--end_idx`: End index in dataset (-1 for all) (default: -1)
- `--