
# **Monocular Depth Estimation with Uncertainty Quantification and Bayesian Machine Learning**

This project applies Bayesian machine learning techniques to monocular depth estimation, using a deep neural network to predict depth maps and quantify uncertainties. 
---
![Aleatoric Uncertainty](<Aleatoric Uncertainty.png>)

![Epistemic Uncertainty](<Epistemic Uncertainty.png>)

---

## **Features**
- **Depth Prediction**: Predict pixel-wise depth using a convolutional neural network.
- **Aleatoric Uncertainty**: Captures data noise using probabilistic outputs.
- **Epistemic Uncertainty**: Uses Monte Carlo (MC) Dropout during inference to estimate model uncertainty.
- **Post-Processing**: Refine depth maps using bilateral filtering for edge preservation.
- **Visualization**: Visualize predicted depth, ground truth, aleatoric uncertainty, and refined maps.

---

## **Model Overview**
- **Encoder**: A ResNet-18 backbone extracts hierarchical features from the input RGB image.
- **Decoder**: Upsamples features to generate depth predictions and log-variance.
- **Probabilistic Loss**:
  \[
  L = 0.5 \cdot 	ext{exp}(-\log(\sigma^2)) \cdot (y - \hat{y})^2 + 0.5 \cdot \log(\sigma^2)
  \]

---

## **Project Workflow**

### **1. Dataset Preparation**
- **Dataset**: NYU Depth V2.
- **Augmentations**:
  - Horizontal flips, Gaussian noise, cropping, resizing, brightness/contrast adjustment, and color jitter.
- **Preprocessing**: Resize images and depth maps to \(224 	imes 224\).

### **2. Training**
- **Optimizer**: AdamW with learning rate \(1e^{-4}\).
- **Training**:
  - Mixed precision training with PyTorch’s `GradScaler`.
  - Fine-tuning with a reduced learning rate \(1e^{-5}\) for 3 epochs.

### **3. Inference**
- **MC Dropout**:
  - Multiple forward passes with active dropout to estimate epistemic uncertainty.
- **Bilateral Filtering**:
  - Smoothens depth predictions while preserving edges.

### **4. Visualization**
- Outputs include:
  - Predicted depth.
  - Ground truth depth.
  - Aleatoric uncertainty.
  - Refined depth map.

---


## **Results**s
### **Outputs**
For a given RGB input, the model produces:
1. **Predicted Depth**: Raw dense depth map.
2. **Refined Depth**: Post-processed depth map.
3. **Aleatoric Uncertainty**: Captures data noise.
4. **Epistemic Uncertainty**: Captures model uncertainty via MC Dropout.

---

## **Future Work**
- **Improve Model Architecture**:
  - Explore more powerful backbones like EfficientNet or Swin Transformer.
- **Generalization**:
  - Extend the model to outdoor datasets like KITTI Depth.
- **Advanced Post-Processing**:
  - Use Conditional Random Fields (CRF) for better alignment of depth predictions.

---

## **Acknowledgements**
- **Datasets**: NYU Depth V2.
- **Libraries**: PyTorch, Albumentations, Segmentation Models PyTorch.

---


