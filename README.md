# Documentation

## Project Introduction

ASCNet is a dual\-branch multimodal fusion deep learning framework for automated glaucoma grading\. The model leverages both **color fundus photographs \(CFP\)** and **optical coherence tomography \(OCT\)** images to achieve three\-category glaucoma diagnosis:

- **Class 0**: Non\-Glaucoma

- **Class 1**: Early Glaucoma

- **Class 2**: Mid\-Advanced Glaucoma

## Model Architecture

### Overall Pipeline

### Core Modules

#### 1\. CSAM \(Circular Structure Adaptation Module\)

The Circular Structure Adaptation Module is designed to strengthen spatial attention perception for fundus images:

- **Channel Attention**: Channel weight learning based on global average pooling and max pooling\.

- **Spatial Attention**: A multi\-radius soft circular kernel design with three parallel branches to learn circular structural patterns of different radii\.

- It better captures near\-circular structures such as the optic disc and optic cup in fundus images\.

#### 2\. LPEAM \(Layer\-Prior Enhanced Anisotropic Attention Module\)

The Layer\-Prior Enhanced Anisotropic Attention Module is specially tailored for OCT images:

- **H\-axis branch**: Axial attention with layer prior bias, utilizing k\_h×1 strip convolutions to compute layer\-structure\-aware weights from the raw feature map\.

- **W\-axis branch**: Standard axial attention mechanism\.

- It effectively models structural features of retinal layers in OCT scans\.

#### 3\. CACSSFM \(Cross\-Modal Asymmetric Conditioned State Space Fusion Module\)

The Cross\-Modal Asymmetric Conditioned State Space Fusion Module achieves deep feature fusion between CFP and OCT:

- **FiLM Conditioning**: Modulation parameters \(γ, β\) generated from the global OCT embedding\.

- **Selective Scan**: Bidirectional \(row\-wise \+ column\-wise\) selective scan based on the 2D state space model\.

- **Dynamic Parameters**: Conditional time step Δdt and state matrix A\.

## Datasets

The GAMMA challenge dataset is adopted, including:

- **Training set**: `dataset/Train/`

- **Validation set**: `dataset/Validation/`

- **Test set**: `dataset/Test/`

Each patient is equipped with:

- 1 color fundus photograph \(CFP\)

- Multiple OCT B\-scan images

## Training Configuration

|Parameter|Value|
|---|---|
|Batch Size|4|
|Epochs|50|
|Initial learning rate|0\.0001|
|Optimizer|Adam|
|Learning rate scheduler|ExponentialLR \(γ=0\.95\)|
|Loss function|Focal Loss \(γ=2, weighted\)|
|Image size|512×512|
|Number of OCT slices|16|

### Data Augmentation

**Fundus images \(CFP\)**:

- Random horizontal flip

- Random rotation \(±30°\)

- Center crop \(512\)

**OCT images**:

- Random horizontal flip

- Random affine transformation \(translation ±5%, scaling 0\.94–1\.06\)

## Evaluation Metrics

The following metrics are automatically calculated after training:

- **Accuracy**: Classification accuracy

- **Kappa**: Cohen’s weighted Kappa

- **Macro\-F1**: Macro\-averaged F1 score

- **Weighted\-F1**: Weighted F1 score

- **Macro\-Recall / Precision / Specificity**: Macro\-averaged recall, precision and specificity

- **Macro\-AUC**: Macro\-averaged ROC\-AUC

## Visualization Analysis

The program automatically generates the following outputs after training:

1. **ROC curve** \(`roc_curve.png`\): ROC curves for each category and macro\-averaged AUC

2. **Confusion matrix** \(`confusion_matrix.png`\): Heatmap of classification results

3. **T\-SNE visualization** \(`tsne_fusion_visualization.png`\): 2D visualization of fused features

4. **GradCAM heatmap** \(optional\): Attention visualization for the CFP branch, OCT branch and fusion module

## Quick Start

### Environment Requirements

```bash
pip install torch torchvision
pip install pytorch-grad-cam
pip install einops
pip install scikit-learn matplotlib pandas
```

### Training

```bash
# Assign GPU
export CUDA_VISIBLE_DEVICES=6

# Run training script
python MCLC_k=5.py

# Or specify the log directory
export LOG_DIR=/path/to/log_directory
python MCLC_k=5.py
```

## Project Structure

```plain
GAMMA/code/
├── MCLC_k=5.py              # Main training script
├── MODEL/
│   ├── Attention.py          # Definition of CSAM module
│   ├── AX.py                 # Definition of LPEAM module
│   ├── Fusion.py             # Definition of CACSSFM fusion module
│   ├── dataloaderMulti.py    # Multimodal data loader
│   └── loss.py               # Focal Loss implementation
└── result/                   # Training logs and visualization outputs
    └── MCLC_k_5/
        ├── training_log.txt
        ├── best_model_*.pth
        ├── roc_curve.png
        ├── confusion_matrix.png
        └── tsne_fusion_visualization.png
```

## Technical Highlights

We propose ASCNet, an asymmetric multimodal fusion network for glaucoma grading\. CACSSFM leverages OCT semantic information to dynamically modulate spatial features derived from CFP\. CSAM extracts near\-circular optic disc and optic cup features via multi\-radius soft convolutions\. LPEAM captures OCT retinal layer pathological characteristics with enhanced axial attention and structural priors\. ASCNet achieves superior performance against state\-of\-the\-art methods on four glaucoma datasets in terms of both accuracy and computational efficiency\.

## Citation

If you find this work useful for your research, please cite our paper\.

