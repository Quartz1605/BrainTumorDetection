# Brain Tumor Detection using Deep Learning

## 📋 Project Overview

This project implements a **binary classification system** for detecting brain tumors in MRI scans using **deep learning techniques**. The model leverages the **VGG-16 architecture** with transfer learning to classify brain MRI images into two categories:
- **NO** - No tumor detected (encoded as `0`)
- **YES** - Tumor detected (encoded as `1`)

### 🎯 Performance Metrics

The model achieves the following accuracy scores:

| Dataset | Accuracy |
|:-------:|:--------:|
| Validation Set | ~86% |
| Test Set | ~82% |

**Accuracy Formula:**
$$\text{Accuracy} = \frac{\text{Number of correctly predicted images}}{\text{Total number of tested images}} \times 100\%$$

---

## 🛠️ Technical Stack

### Libraries and Dependencies

```python
# Core Libraries
import numpy as np
import cv2
import os
import shutil
import itertools
import imutils

# Data Science and ML
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Visualization
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

# Deep Learning (Keras/TensorFlow)
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping

# Progress tracking
from tqdm import tqdm
```

### 📦 Installation Requirements

```bash
pip install numpy opencv-python scikit-learn matplotlib plotly keras tensorflow imutils tqdm
```

---

## 📊 Dataset Structure

### Initial Dataset Organization
```
brain_tumor_dataset/
├── yes/           # Images with tumors
└── no/            # Images without tumors
```

### Project Data Split Strategy

The notebook automatically reorganizes the data into train/validation/test splits:

```
Project Structure After Split:
├── TRAIN/
│   ├── YES/       # Training images with tumors
│   └── NO/        # Training images without tumors
├── VAL/
│   ├── YES/       # Validation images with tumors
│   └── NO/        # Validation images without tumors
├── TEST/
│   ├── YES/       # Test images with tumors
│   └── NO/        # Test images without tumors
├── TRAIN_CROP/    # Cropped training images
├── VAL_CROP/      # Cropped validation images
└── TEST_CROP/     # Cropped test images
```

### Data Split Logic
- **First 5 images** of each class → Test set
- **Next 80%** of remaining images → Training set  
- **Last 20%** → Validation set

---

## 🔧 Data Preprocessing Pipeline

### 1. Image Loading
```python
def load_data(dir_path, img_size=(100,100)):
    """
    Load resized images as np.arrays to workspace
    """
```
- Loads images from specified directory
- Converts to numpy arrays
- Assigns labels (0 for NO, 1 for YES)
- Returns: X (images), y (labels), labels dictionary

### 2. Brain Extraction and Cropping
```python
def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
```

**Cropping Process:**
1. **Convert to grayscale** using `cv2.cvtColor()`
2. **Apply Gaussian blur** with (5,5) kernel
3. **Threshold image** at value 45
4. **Apply morphological operations** (erosion → dilation)
5. **Find contours** and select largest one
6. **Extract extreme points** (left, right, top, bottom)
7. **Crop rectangular region** around brain area

### 3. Image Preprocessing for VGG-16
```python
def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
```
- Resizes images to **224×224 pixels** (VGG-16 input size)
- Applies **VGG-16 preprocessing** using `preprocess_input()`
- Uses **INTER_CUBIC** interpolation for high-quality resizing

---

## 🎨 Data Visualization

### Class Distribution Analysis
- **Interactive bar charts** using Plotly
- Shows count of tumor/non-tumor images across train/val/test sets
- Helps identify class imbalance issues

### Sample Image Display
```python
def plot_samples(X, y, labels_dict, n=50):
    """
    Creates a gridplot for desired number of images (n) from the specified set
    """
```

### Image Ratio Analysis
- Analyzes width/height ratios of all images
- Creates histogram distribution
- Helps understand image geometry variations

### Confusion Matrix Visualization
```python
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    """
    Prints and plots the confusion matrix with color coding
    """
```

---

## 🔄 Data Augmentation Strategy

### Augmentation Parameters
```python
demo_datagen = ImageDataGenerator(
    rotation_range=15,           # Random rotation ±15°
    width_shift_range=0.05,      # Horizontal shift ±5%
    height_shift_range=0.05,     # Vertical shift ±5%
    rescale=1./255,              # Normalize pixel values [0,1]
    shear_range=0.05,            # Shear transformation ±5%
    brightness_range=[0.1, 1.5], # Brightness variation 10%-150%
    horizontal_flip=True,         # Random horizontal flip
    vertical_flip=True           # Random vertical flip
)
```

### Training vs Validation Generators
- **Training generator**: Applies full augmentation
- **Validation generator**: Only applies preprocessing (no augmentation)

---

## 🧠 Model Architecture

### Transfer Learning with VGG-16

```python
# Load pre-trained VGG-16 (without top classification layer)
base_model = VGG16(
    weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    include_top=False, 
    input_shape=(224, 224, 3)
)

# Custom classification head
model = Sequential([
    base_model,                           # VGG-16 feature extractor
    layers.Flatten(),                     # Flatten feature maps
    layers.Dropout(0.5),                  # Regularization (50% dropout)
    layers.Dense(1, activation='sigmoid') # Binary classification
])
```

### Model Configuration
- **Base Model**: Pre-trained VGG-16 (frozen weights)
- **Classification Head**: Single dense layer with sigmoid activation
- **Optimizer**: RMSprop (learning rate = 1e-4)
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy

### Training Configuration
- **Epochs**: 30 (with early stopping)
- **Batch Size**: 32 (training), 16 (validation)
- **Steps per Epoch**: 50
- **Validation Steps**: 25
- **Early Stopping**: Monitors validation accuracy with patience=6

---

## 📈 Model Performance Analysis

### Training Monitoring
- **Accuracy curves** for train and validation sets
- **Loss curves** for train and validation sets
- **Early stopping** to prevent overfitting

### Evaluation Metrics
1. **Accuracy Score**: Primary metric for model evaluation
2. **Confusion Matrix**: Detailed classification breakdown
3. **Misclassification Analysis**: Visual inspection of incorrectly classified images

### Performance Visualization
```python
# Plotting training history
plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Val Set')
plt.title('Model Loss')
```

---

## 🚀 Usage Instructions

### 1. Environment Setup
```bash
# Install required packages
pip install imutils numpy opencv-python matplotlib scikit-learn keras tensorflow plotly tqdm

# For Jupyter notebook
pip install jupyter ipywidgets
```

### 2. Data Preparation
```python
# Ensure your dataset follows this structure:
brain_tumor_dataset/
├── yes/    # Tumor images
└── no/     # Non-tumor images
```

### 3. Running the Notebook
1. **Open Jupyter notebook**: `brain-tumor-detection.ipynb`
2. **Execute cells sequentially** from top to bottom
3. **Monitor training progress** through plots and metrics
4. **Evaluate results** on test set

### 4. Model Prediction
```python
# For new images
predictions = model.predict(preprocessed_images)
binary_predictions = [1 if x > 0.5 else 0 for x in predictions]
```

---

## 📁 File Structure

```
brain-tumor/
├── brain-tumor-detection.ipynb    # Main notebook
├── README.md                      # This documentation
├── TRAIN/                         # Training data
├── VAL/                          # Validation data
├── TEST/                         # Test data
├── TRAIN_CROP/                   # Cropped training images
├── VAL_CROP/                     # Cropped validation images
├── TEST_CROP/                    # Cropped test images
└── preview/                      # Temporary augmentation preview
```

---

## 🔬 Technical Details

### Image Processing Pipeline
1. **Loading**: Images loaded as RGB arrays
2. **Cropping**: Brain region extraction using contour detection
3. **Resizing**: Standardized to 224×224 pixels
4. **Normalization**: VGG-16 specific preprocessing
5. **Augmentation**: Real-time data augmentation during training

### Model Training Process
1. **Transfer Learning**: Use pre-trained VGG-16 features
2. **Feature Extraction**: Freeze base model weights
3. **Custom Head**: Train only classification layers
4. **Regularization**: Dropout layer prevents overfitting
5. **Early Stopping**: Automatic training termination

### Evaluation Strategy
1. **Train/Val Split**: Monitor overfitting during training
2. **Test Set Evaluation**: Final unbiased performance assessment
3. **Confusion Matrix**: Detailed classification analysis
4. **Error Analysis**: Visual inspection of misclassified cases

---

## 🎯 Model Limitations and Future Improvements

### Current Limitations
- **Dataset Size**: Relatively small dataset may limit generalization
- **Class Imbalance**: Potential imbalance between tumor/non-tumor images
- **Single Architecture**: Only VGG-16 tested
- **Binary Classification**: Doesn't classify tumor types

### Potential Improvements
1. **Data Enhancement**:
   - Larger, more diverse dataset
   - Multi-institutional data collection
   - Better class balancing techniques

2. **Model Architecture**:
   - Try newer architectures (ResNet, DenseNet, EfficientNet)
   - Ensemble methods
   - Custom CNN architectures

3. **Advanced Techniques**:
   - Grad-CAM for visualization
   - Multi-class classification (tumor types)
   - 3D CNN for volumetric analysis
   - Attention mechanisms

4. **Preprocessing Enhancements**:
   - Advanced brain extraction algorithms
   - Intensity normalization
   - Histogram equalization

---

## 🔍 Code Structure Analysis

### Key Functions

| Function | Purpose | Input | Output |
|----------|---------|-------|---------|
| `load_data()` | Load and organize image data | Directory path, image size | Arrays (X, y), labels dict |
| `crop_imgs()` | Extract brain region from MRI | Image set, padding pixels | Cropped image array |
| `preprocess_imgs()` | VGG-16 preprocessing | Image set, target size | Preprocessed array |
| `plot_samples()` | Visualize image samples | Images, labels, count | Matplotlib plots |
| `plot_confusion_matrix()` | Display confusion matrix | CM matrix, class names | Formatted plot |
| `save_new_images()` | Save processed images | Image set, labels, folder | Saved files |

### Configuration Variables

```python
# Global Configuration
RANDOM_SEED = 123           # Reproducibility
IMG_SIZE = (224, 224)       # VGG-16 input size
NUM_CLASSES = 1             # Binary classification
EPOCHS = 30                 # Maximum training epochs

# Directory Paths
IMG_PATH = '../input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/'
TRAIN_DIR = 'TRAIN/'
TEST_DIR = 'TEST/'
VAL_DIR = 'VAL/'
```

---

## 📊 Results Summary

### Model Performance
- **Validation Accuracy**: ~86%
- **Test Accuracy**: ~82%
- **Training Strategy**: Transfer learning with VGG-16
- **Regularization**: Dropout (0.5) + Early stopping

### Dataset Statistics
- **Training Images**: Majority of dataset (~75%)
- **Validation Images**: ~20% of dataset
- **Test Images**: First 5 images of each class
- **Image Size**: 224×224 pixels (RGB)

### Processing Pipeline
- **Brain Extraction**: Automated using contour detection
- **Data Augmentation**: 8 different transformations
- **Preprocessing**: VGG-16 compatible normalization
- **Batch Processing**: Efficient memory usage

---

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Test thoroughly
5. Submit pull request

### Suggested Contributions
- Additional model architectures
- Advanced preprocessing techniques
- Performance optimization
- Enhanced visualization
- Documentation improvements

---

## 📞 Support

For questions, issues, or suggestions:
- Create an issue in the repository
- Review the notebook documentation
- Check the confusion matrix for model insights
- Examine misclassified examples for debugging

---

## 📜 License

This project is for educational and research purposes. Please ensure compliance with medical imaging regulations and data privacy laws when using real medical data.

---

*This README provides comprehensive documentation for the brain tumor detection project. For technical details, refer to the Jupyter notebook implementation.*
