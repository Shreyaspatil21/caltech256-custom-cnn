# Caltech-256 Image Classification using a Custom CNN

## Author
Shreyas Patil

## Project Overview
This project implements a Convolutional Neural Network (CNN) built from scratch to classify images from the Caltech-256 dataset.  
No pretrained models or transfer learning were used, as per assignment requirements.

The goal is to preprocess the dataset, build a deep CNN architecture, train it from scratch, evaluate its performance, and analyze the results using plots and a confusion matrix.

---

## Dataset Description
**Dataset:** Caltech-256  
**Classes:** 256  
**Total Images:** 30,607  

The dataset contains images of everyday objects with large variations in viewpoint, lighting, scale, and background.  
For this project, the dataset was downloaded through KaggleHub and manually split into:

- **Training:** 21,308 images  
- **Validation:** 4,475 images  
- **Test:** 4,825 images  

Each image was resized to 224×224 and normalized using ImageNet statistics.

---

## Model Summary
A custom VGG-style CNN was designed and implemented.  
The architecture includes:

### Convolutional Blocks
- **Block 1:**  
  - Conv (64 filters)  
  - Conv (64 filters)  
  - Max Pool  

- **Block 2:**  
  - Conv (128 filters)  
  - Conv (128 filters)  
  - Max Pool  

- **Block 3:**  
  - Conv (256 filters)  
  - Conv (256 filters)  
  - Conv (256 filters)  
  - Max Pool  

- **Block 4:**  
  - Conv (512 filters)  
  - Conv (512 filters)  
  - Conv (512 filters)  
  - Max Pool  

### Fully Connected Layers
- Linear (4096 units) + ReLU + Dropout  
- Linear (4096 units) + ReLU + Dropout  
- Output Layer (256 units for 256 classes)

### Training Details
- Optimizer: Adam  
- Loss Function: CrossEntropyLoss  
- Epochs: 20  
- Batch Size: 32  
- Learning Rate Scheduler: StepLR  
- Augmentations: RandomHorizontalFlip  

---

## Results
The custom CNN achieved the following performance:

- **Final Validation Accuracy:** 80.56%  
- **Final Test Accuracy:** 81.21%  

Training and validation curves were generated to analyze the learning behavior.

### Included Artifacts
- `loss_curve.png` — Training vs validation loss  
- `accuracy_curve.png` — Validation accuracy over epochs  
- `sample_predictions.png` — Model predictions on sample test images  
- `confusion_matrix.npy` — Numerical confusion matrix data  
- `notebook.ipynb` — Complete Jupyter/Kaggle Notebook  
- `final_model.pth` — Trained model weights (if under 100MB)  

---

## Observations
1. The model shows steady improvement in both training and validation metrics.  
2. Validation accuracy increases sharply during early epochs and stabilizes after epoch 15.  
3. There is no significant overfitting due to proper augmentation and dropout.  
4. Misclassifications mainly occur in visually similar classes.  
5. The model generalizes well despite being trained from scratch.

---

## Improvements Tried
- Data augmentation (RandomHorizontalFlip)  
- Learning rate scheduler for stable convergence  
- Deeper architecture with additional convolutional layers  
- Dropout in fully connected layers to reduce overfitting  

These improvements contributed to achieving over **80% accuracy** on a very challenging 256-class dataset.

---

## How to Run the Project
1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
