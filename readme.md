# Safety Equipment Detection System ⛑️

A computer vision system for real-time detection and monitoring of safety equipment using YOLO11 deep learning model with <b>93% accuracy </b>. This project aims to enhance workplace safety by automatically identifying and tracking safety gear compliance through images and video streams.

<p align="left">
  <a href="https://drive.google.com/file/d/13V--dpmoqjjwmrTrwi64l5YQeRaIQQFZ/view?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/▶%20Watch%20Demo-Video-red?style=for-the-badge">
  </a>
  <a href="https://github.com/Subho4531/Safety-Object-Detection/blob/master/Safety%20equipment%20Detection%20Report.pdf" target="_blank">
    <img src="https://img.shields.io/badge/Safety%20Equipment%20Detection%20Report-blue?style=for-the-badge">
  </a>
  <a href="https://github.com/Subho4531/Safety-Object-Detection?tab=readme-ov-file#-testing-and-results" target="_blank">
    <img src="https://img.shields.io/badge/Testing%20Reasults%20Report-green?style=for-the-badge">
  </a>
</p>


---

## 📋 Table of Contents
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Dataset](#-dataset)
- [Model Information](#-model-information)
- [Testing and Results](#-testing-and-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)

---

The Safety Equipment Detection System uses state-of-the-art computer vision technology to automatically detect and monitor safety equipment in real-time. The solution:

1. **Accepts multiple input formats**: Images, videos, and live camera feeds
2. **Detects safety equipment** using a trained YOLO11 model
3. **Provides real-time alerts** when safety violations are detected
4. **Generates visual feedback** with bounding boxes and confidence scores
5. **Offers an interactive web interface** built with Streamlit for easy deployment

### System Workflow

```
Input Source → Frame Processing → YOLO11 Detection → 
Bounding Box Generation → Confidence Filtering → 
Visual Output with Annotations → Results Display
```

**Key Components:**
- **Input Processing**: Handles images, videos, and camera streams
- **Detection Engine**: YOLO11 model trained on safety equipment dataset
- **Post-Processing**: Confidence thresholding and label generation
- **Visualization**: Real-time display with detection overlays
- **Web Interface**: Streamlit application for user interaction

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────┐
│   Input Sources     │
│ - Images            │
│ - Videos            │
│ - Live Camera       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Preprocessing      │
│ - Frame Extraction  │
│ - Normalization     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   YOLO11 Model      │
│ - Feature Extract   │
│ - Object Detection  │
│ - Classification    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Post-Processing    │
│ - NMS               │
│ - Confidence Filter │
│ - Bounding Boxes    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Visualization     │
│ - Annotated Output  │
│ - Statistics        │
│ - Alerts            │
└─────────────────────┘
```

### Technical Stack Flow

**Frontend (Streamlit) ↔ Backend (Python) ↔ Model (YOLO11) ↔ Data (Images/Video)**

---

## ✨ Features

### Core Functionality
- ✅ **Multi-format Input Support**: Images (PNG, JPG), Videos (MP4, AVI), and live camera feeds
- ✅ **Real-time Detection**: Process video streams with FPS tracking
- ✅ **Multiple Object Detection**: Simultaneous detection of various safety equipment
- ✅ **Confidence Scoring**: Adjustable threshold for detection accuracy
- ✅ **Interactive Web Interface**: User-friendly Streamlit dashboard

### Detection Capabilities
- Detection of various safety equipment categories
- Bounding box visualization with class labels
- Confidence score display for each detection
- Real-time processing statistics

### User Interface Features
- Adjustable confidence threshold (5% - 95%)
- Toggle options for FPS, labels, and confidence display
- Live camera integration
- Video playback with detection overlays
- Download processed results
- Detection statistics dashboard

---

## 🛠️ Technologies Used

### Deep Learning Framework
- **YOLOv11 (Ultralytics)**: Latest version of YOLO for object detection
  - Fast inference speed
  - High accuracy
  - Efficient architecture
  - Pre-trained weights available

### Programming & Libraries
- **Python 3.8+**: Core programming language
- **OpenCV (cv2)**: Image and video processing
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **Pillow (PIL)**: Image handling
- **PyTorch**: Deep learning backend

### Development Tools
- **Jupyter Notebook**: Model training and experimentation
- **YAML**: Configuration management
- **Git**: Version control

---

## 📊 Dataset

The model is trained on a custom dataset containing images of various safety equipment in different environments and conditions.

### Dataset Characteristics
- **Image Format**: YOLO format annotations
- **Split Structure**:
  - Training set
  - Validation set
  - Test set
- **Annotation Format**: Bounding boxes with class labels
- **Configuration**: Defined in `yolo_params.yaml`

### Data Organization
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 🤖 Model Information

### YOLO11 Architecture
YOLO11 is the latest iteration in the YOLO family, offering:
- **Single-stage detection**: Direct prediction without region proposals
- **Anchor-free design**: Eliminates complex anchor box tuning
- **Efficient backbone**: Optimized feature extraction
- **Multi-scale detection**: Detects objects at various sizes

### Model Training
- **Base Model**: YOLO11m (medium variant)
- **Training Framework**: Ultralytics YOLO
- **Weights**: Custom trained on safety equipment dataset
- **Location**: `models/best.pt`

### Training Configuration
Training parameters are stored in `models/training/runs/train/yolo11m_space_run/args.yaml`

Key configurations include:
- Batch size, learning rate, epochs
- Image size and augmentation parameters
- Optimizer settings
- Loss functions

---

## 📈 Testing and Results

### Model Evaluation

The model was evaluated on a dedicated test set using standard object detection metrics.

#### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP@50** | 0.858| Mean Average Precision at IoU threshold 0.5 |
| **mAP@50-95** | 0.756 | Mean Average Precision across IoU thresholds 0.5 to 0.95 |
| **Precision** | 0.93 | Ratio of true positive detections |
| **Recall** | 0.77 | Ratio of detected objects to ground truth |
| **Inference Speed** | 14.5 | Frames processed per second |

*Note: Actual metrics can be found in `models/training/runs/train/yolo11m_space_run/results.csv`*

### Testing Methodology

1. **Test Set Evaluation**
   - Model tested on unseen images
   - Predictions saved in `testing_results/runs/detect/val/`
   - Bounding boxes and labels generated

2. **Confidence Threshold Testing**
   - Default threshold: 0.25
   - Tested range: 0.05 - 0.95
   - Optimal threshold determined through validation

3. **Real-world Testing**
   - Video stream processing
   - Live camera feed testing
   - FPS performance monitoring

### Prediction System

The `predict.py` script provides automated testing:
- Loads test images from configured directory
- Generates predictions with bounding boxes
- Saves annotated images and label files
- Calculates validation metrics

**Output Structure:**
```
predictions/
├── images/          # Annotated images
└── labels/          # Detection coordinates (YOLO format)
```

### Test Results Observations

✅ **Strengths:**
- High accuracy on well-lit images
- Robust detection across different angles
- Fast inference suitable for real-time applications
- Low false positive rate

⚠️ **Challenges:**
- Performance may vary in low-light conditions
- Occlusion can affect detection accuracy
- Small objects at distance may be missed

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (optional, for live detection)
- Git (for cloning repository)

### Setup Instructions

1. **Clone the Repository**
```bash
git clone <repository-url>
cd "Safety Object Detection"
```

2. **Create Virtual Environment** (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify Model Files**
Ensure `models/best.pt` exists. If not, place your trained YOLO11 weights in this location.

5. **Configure Dataset Paths** (for testing)
Edit `testing_results/yolo_params.yaml` to point to your test data directory.

---

## 💻 Usage

### Web Application (Streamlit)

**Launch the web interface:**
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

**Using the Interface:**
1. Select input source from the sidebar (Image/Video/Live Camera)
2. Adjust confidence threshold as needed
3. Toggle display options (FPS, Labels, Confidence)
4. Upload file or start camera
5. View real-time detections with bounding boxes
6. Download processed results

### Prediction Script

**Run automated predictions:**
```bash
cd testing_results
python predict.py
```

This will:
- Load test images from the configured directory
- Apply the trained model
- Save predictions in `predictions/` folder
- Generate validation metrics

### Training (Advanced)

To retrain or fine-tune the model, refer to the Jupyter notebook:
```
models/training/training-notebook.ipynb
```

---

## 📁 Project Structure

```
Safety Object Detection/
│
├── app.py                          # Streamlit web application
├── readme.md                       # Project documentation
├── requirements.txt                # Python dependencies
│
├── models/
│   ├── best.pt                     # Trained YOLO11 weights
│   └── training/
│       ├── training-notebook.ipynb # Model training notebook
│       └── runs/
│           └── train/
│               └── yolo11m_space_run/
│                   ├── args.yaml        # Training arguments
│                   ├── results.csv      # Training metrics
│                   └── weights/
│                       ├── best.pt      # Best model checkpoint
│                       └── last.pt      # Last epoch checkpoint
│
└── testing_results/
    ├── best.pt                     # Test model weights
    ├── classes.txt                 # Class names
    ├── predict.py                  # Prediction script
    ├── yolo_params.yaml            # Test configuration
    └── runs/
        └── detect/
            └── val/                # Validation results
```

---

## ⚠️ Limitations

### Current Constraints

1. **Environmental Factors**
   - Performance degrades in poor lighting conditions
   - Heavily occluded objects may not be detected
   - Extreme angles can affect accuracy

2. **Hardware Requirements**
   - Real-time processing requires decent GPU
   - High-resolution video may strain CPU-only systems
   - Webcam quality affects live detection accuracy

3. **Model Limitations**
   - Limited to safety equipment classes in training data
   - May not generalize to unseen equipment types
   - Confidence scores vary based on image quality

4. **Technical Constraints**
   - Fixed model architecture (YOLO11m)
   - Requires proper lighting for optimal results
   - Internet connection needed for initial model download

---

## 🔮 Future Improvements

### Planned Enhancements

1. **Model Improvements**
   - [ ] Train on larger, more diverse dataset
   - [ ] Implement ensemble methods for higher accuracy
   - [ ] Add support for smaller YOLO variants for edge devices
   - [ ] Fine-tune for specific industry use cases

2. **Feature Additions**
   - [ ] Multi-camera support and synchronization
   - [ ] Alert system for safety violations
   - [ ] Database integration for logging detections
   - [ ] Email/SMS notifications for violations
   - [ ] Automated report generation

3. **Performance Optimization**
   - [ ] Model quantization for faster inference
   - [ ] TensorRT optimization for NVIDIA GPUs
   - [ ] ONNX export for cross-platform deployment
   - [ ] Batch processing for multiple videos

4. **User Experience**
   - [ ] Mobile application development
   - [ ] Cloud deployment (AWS/Azure/GCP)
   - [ ] User authentication and access control
   - [ ] Customizable detection zones
   - [ ] Historical data analytics dashboard

5. **Integration Capabilities**
   - [ ] REST API for third-party integration
   - [ ] Webhook support for real-time alerts
   - [ ] Integration with existing security systems
   - [ ] Export results to various formats (JSON, CSV, XML)

---

## 📄 License

This project is intended for educational and workplace safety purposes. Please ensure compliance with local regulations when deploying in production environments.

---

## 👥 Contributing

Contributions are welcome! Areas for contribution:
- Dataset expansion and annotation
- Model performance improvements
- UI/UX enhancements
- Documentation and tutorials
- Bug fixes and optimization

---

## 📞 Support

For questions, issues, or collaboration:
- Open an issue in the repository
- Check existing documentation
- Review training notebooks for technical details

---

## 🙏 Acknowledgments

- Ultralytics team for the YOLO framework
- Streamlit for the web framework
- Open-source computer vision community

---

## 📊 Quick Start Example

```python
# Load model and make a prediction
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO('models/best.pt')

# Predict on an image
results = model.predict('path/to/image.jpg', conf=0.25)

# Display results
annotated_img = results[0].plot()
cv2.imshow('Detection', annotated_img)
cv2.waitKey(0)
```

---

**Built with ❤️ for workplace safety**
