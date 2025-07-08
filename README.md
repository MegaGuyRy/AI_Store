# AI Store

### Overview
This project is an AI-powered computer vision system that classifies store products (like sauce, noodles, and powder) from images or a live webcam feed. The goal is to automate product recognition using convolutional neural networks (CNNs), making it easier to track or identify items in a store setting.

Built with TensorFlow and OpenCV, the model is trained on a custom dataset of labeled product images. It uses ResNet50 for robust feature extraction and real-time classification.

### Features
- Classifies products into categories 
- Supports predictions on images or live webcam video  
- Built using TensorFlow with EfficientNet for high accuracy  
- Includes data preprocessing pipeline to clean and prepare input images  
- Outputs predicted class and confidence scores

### Dataset
The dataset consists of:
- ~30 annotated images for training and validation (10 images per class: sauce, noodles, powder)
- Images were labeled using [LabelImg](https://github.com/heartexlabs/labelImg) and converted into a format compatible with TensorFlow pipelines.

### Model Architecture
- **Feature extractor:** EfficientNetB0 (transfer learning)  
- **Classifier:** Fully connected dense layers on top  
- Trained using Adam optimizer, categorical crossentropy loss  
- Early stopping and model checkpointing to avoid overfitting
