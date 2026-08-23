# Model Training

This directory contains the training scripts and configuration used to train the crop disease classification models for this project.

The project uses **TensorFlow/Keras** to train crop-specific image classification models. Separate models can be trained for different crops and their corresponding disease classes.

## Crops

The training pipeline supports the following crop datasets:

* Corn
* Cotton
* Paddy
* Sugarcane
* Wheat

Each crop dataset contains images organized according to their respective disease or healthy classes.

## Training Pipeline

The general training workflow is:

```text
Dataset
   ↓
Dataset Validation
   ↓
Train / Validation Split
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
Class Weight Calculation
   ↓
CNN / Transfer Learning Model
   ↓
Model Training
   ↓
Validation
   ↓
Performance Evaluation
   ↓
Save .keras Model
```

## Dataset Structure

The datasets are organized by crop and disease class.

Example:

```text
datasets/
├── corn dataset/
│   ├── Corn_Healthy/
│   ├── Corn_CommonRust/
│   └── Corn_LeafBlight/
│
├── cotton dataset/
│   ├── Cotton_Healthy/
│   └── ...
│
├── paddy dataset/
│   ├── Paddy_BacterialLeafBlight/
│   ├── Paddy_BacterialLeafStreak/
│   ├── Paddy_BacterialPanicleBlight/
│   ├── Paddy_Blast/
│   ├── Paddy_BrownSpot/
│   ├── Paddy_Healthy/
│   └── Paddy_Tungro/
│
├── sugarcane dataset/
│   ├── Sugarcane_BacterialBlight/
│   ├── Sugarcane_Healthy/
│   └── Sugarcane_RedRot/
│
└── wheat dataset/
    └── ...
```

## Preprocessing

Before training, images are:

1. Loaded from the crop-specific dataset.
2. Resized to the input resolution required by the model.
3. Normalized according to the model preprocessing requirements.
4. Augmented during training to improve generalization.

Typical augmentation techniques include:

* Random rotation
* Horizontal flipping
* Zoom
* Translation
* Small changes in image appearance

Augmentation is applied only to the training data and not to the validation/test data.

## Class Weights

Class weights are used when necessary to reduce the effect of class imbalance.

The weights are calculated from the training distribution rather than manually changing the importance of individual disease classes.

This helps prevent the model from becoming biased toward classes containing more images.

## Model Output

Each trained model produces a probability for every class it was trained on.

The class with the highest confidence is selected as the predicted disease/health condition.

Example:

```text
Input Image
     ↓
Crop-specific Model
     ↓
Class Probabilities
     ↓
Highest Confidence Class
     ↓
Disease Prediction
```

## Unknown Disease Handling

The inference system is designed so that a prediction should not automatically be treated as a known disease when the input does not sufficiently match the trained classes.

A confidence threshold can be used during inference.

If the prediction confidence is below the selected threshold, the system can return:

```text
Unknown Disease
```

This prevents the system from confidently assigning an unsupported disease class to an unfamiliar image.

## Training Environment

The models can be trained locally using:

* Python
* TensorFlow
* Keras
* NumPy
* OpenCV
* scikit-learn

GPU acceleration can be used when a compatible NVIDIA GPU and CUDA-enabled TensorFlow environment are available.

## Training

A typical training command is:

```bash
python train.py
```

For a crop-specific training script, the crop can be selected according to the implementation, for example:

```bash
python train.py --crop corn
```

or:

```bash
python train.py --crop sugarcane
```

The exact command depends on the training script used in this repository.

## Saved Models

After training, the final models are saved in Keras format:

```text
.keras
```

Example:

```text
models/
├── corn.keras
├── cotton.keras
├── paddy.keras
├── sugarcane.keras
└── wheat.keras
```

The saved `.keras` models can be loaded directly for local inference without requiring the original training dataset.

## Evaluation

The trained models should be evaluated using data that was not used for model optimization.

Important evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

The confusion matrix is particularly useful for identifying diseases that the model frequently confuses with one another.

## Reproducibility

For reproducible training, record:

* Dataset version
* Number of images per class
* Image resolution
* Train/validation split
* Batch size
* Number of epochs
* Learning rate
* Model architecture
* Data augmentation settings
* Class-weight configuration
* TensorFlow/Keras version

This documentation allows the trained models to be reproduced or improved in future versions of the project.

## Training vs Inference

The training dataset is required for **training**, but it is not required to perform inference using an already trained `.keras` model.

```text
TRAINING
Dataset → Training → .keras Model

INFERENCE
Camera Image → .keras Model → Prediction
```

This separation allows the final application to run locally using the trained model without loading the complete multi-gigabyte training dataset.
