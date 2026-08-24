# CropAI Model Training

This directory contains the training resources and trained AI models used by **CropAI – Multi-Crop AI Farm Assist**.

The project uses crop-specific disease classification models trained on leaf-image datasets. Separate models are used for Corn, Cotton, Paddy, Sugarcane, and Wheat.

---

## Training Directory

```text
training/
│
├── README.md
│
└── keras models/
    ├── corn_disease_best.keras
    ├── cotton_disease_v8_best.keras
    ├── paddy_disease_v2_best.keras
    ├── sugarcane_disease_v2_best.keras
    └── wheat_disease_best.keras

Supported Crops

The current system contains five trained crop models:

Crop	Model
Corn	corn_disease_best.keras
Cotton	cotton_disease_v8_best.keras
Paddy	paddy_disease_v2_best.keras
Sugarcane	sugarcane_disease_v2_best.keras
Wheat	wheat_disease_best.keras
Dataset Structure

The datasets used for training are stored in the repository under:

datasets/
├── corn dataset/
├── cotton dataset/
├── paddy dataset/
├── sugarcane dataset/
└── wheat dataset/

Each crop dataset is organized into class folders.

The class folders represent the diseases and healthy condition used by the corresponding crop model.

Crop Disease Classes
Corn

The Corn model is trained for the available Corn disease/healthy classes, including:

Common Rust
Gray Leaf Spot
Healthy
Leaf Blight
Cotton

The Cotton dataset contains crop-specific disease and healthy classes used by the trained Cotton model.

Paddy

The Paddy dataset contains:

Bacterial Leaf Blight
Bacterial Leaf Streak
Bacterial Panicle Blight
Blast
Brown Spot
Healthy
Tungro
Sugarcane

The Sugarcane dataset contains:

Bacterial Blight
Healthy
Red Rot
Wheat

The Wheat dataset contains the disease and healthy classes used by the Wheat model.

Training Script

The main training script is located at:

Scripts/
└── crop_training_script.py

This script provides the training workflow used to create the crop disease classification models.

The training process includes:

Loading images from class directories
Preparing the dataset
Image preprocessing
Data augmentation
Model training
Validation
Monitoring model performance
Saving the best trained model
Image Preprocessing

The input leaf images are prepared before being supplied to the neural network.

The preprocessing workflow includes:

Loading images from the dataset
Reading and validating image files
Resizing images to the required model input size
Converting images into numerical tensors
Applying the required normalization/scaling

The preprocessing used during inference must remain compatible with the preprocessing used during training.

Data Augmentation

Data augmentation is used to improve model generalization.

Real agricultural images can vary because of:

Leaf orientation
Camera position
Lighting conditions
Backgrounds
Distance from the camera
Image framing

Augmentation can introduce controlled variations such as:

Rotation
Horizontal flipping
Zoom
Translation
Other image transformations configured in the training pipeline

The original dataset images are not permanently modified.

Crop-Specific Models

Instead of training one classifier containing every disease from every crop, CropAI uses separate models.

                       Leaf Image
                           │
                           ▼
                    Crop Identification
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
        Corn            Cotton            Paddy
          │                │                │
          ▼                ▼                ▼
      Corn Model       Cotton Model      Paddy Model
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                   Disease Prediction

The same approach is used for Sugarcane and Wheat.

This allows each model to focus on the disease characteristics of its own crop.

Model Training and Validation

The dataset is divided into training and validation data according to the training configuration.

The training set is used to learn disease patterns.

The validation set is used to monitor how well the model performs on images that were not directly used to update the model weights.

Validation helps identify:

Overfitting
Underfitting
Poor class separation
Training instability
Weak disease classes
Best Model

The project saves the best-performing model during training rather than simply using the model from the final epoch.

The resulting files use the Keras format:

.keras

Current trained models:

keras models/
├── corn_disease_best.keras
├── cotton_disease_v8_best.keras
├── paddy_disease_v2_best.keras
├── sugarcane_disease_v2_best.keras
└── wheat_disease_best.keras
Model Inference

After training, the .keras models are used by the CropAI application.

The inference workflow is:

Camera / Uploaded Image
          │
          ▼
     Image Processing
          │
          ▼
    Crop Identification
          │
          ▼
   Crop-Specific Model
          │
          ▼
   Disease Classification
          │
          ▼
    Confidence Score
          │
          ▼
 Disease / Healthy Result

The application can use images captured through the camera as well as uploaded images.

Multi-View Prediction

CropAI can use multiple views of a leaf during analysis.

Multiple predictions can be combined to improve the stability of the final result.

                    Leaf
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      View 1       View 2       View 3
        │            │            │
        ▼            ▼            ▼
    Prediction   Prediction   Prediction
        │            │            │
        └────────────┼────────────┘
                     ▼
             Prediction Fusion
                     │
                     ▼
              Final Prediction

This is useful when a single image does not provide a complete view of the leaf.

Confidence-Based Routing

The system uses prediction confidence as part of the analysis workflow.

A low-confidence prediction should not automatically be treated as a confirmed disease.

The system can use the confidence result to identify uncertain predictions and request a clearer image when required.

Factors that can reduce prediction reliability include:

Poor lighting
Blurred images
Partial leaves
Complex backgrounds
Unusual symptoms
Diseases not represented in the training dataset
Incorrect crop selection
Training to Deployment

The complete workflow is:

Dataset
   │
   ▼
Training Script
   │
   ▼
Crop-Specific Model
   │
   ▼
Best .keras Model
   │
   ▼
CropAI Application
   │
   ▼
Camera / Image Upload
   │
   ▼
Disease Analysis

The trained models are therefore the connection between the training pipeline and the final CropAI application.

Git LFS

The datasets and trained Keras models contain large binary files.

Git Large File Storage (Git LFS) is used to manage these files.

To view LFS-managed files:

git lfs ls-files

The .gitattributes file contains the Git LFS tracking configuration.

Reproducing Training

To reproduce or modify the training process, use:

datasets/
Scripts/crop_training_script.py

The training environment should contain the required Python and machine-learning dependencies.

Important training parameters include:

Dataset path
Class names
Image size
Batch size
Number of epochs
Learning rate
Data augmentation
Validation configuration
Model architecture
Optimizer
Loss function
Checkpoint configuration

Changing these parameters can produce a different trained model.

Model Usage

The trained models are intended to be loaded by the CropAI inference application.

The model files should remain in:

training/keras models/

when the repository is used as the reference project.

If the application expects a different model path, the corresponding path in the application configuration should be updated.

Current Training Resources
Resource	Status
Corn dataset	Available
Cotton dataset	Available
Paddy dataset	Available
Sugarcane dataset	Available
Wheat dataset	Available
Training script	Available
Corn Keras model	Available
Cotton Keras model	Available
Paddy Keras model	Available
Sugarcane Keras model	Available
Wheat Keras model	Available
Git LFS configuration	Available
Future Improvements

Future versions of the training pipeline can include:

Larger field-collected datasets
More diverse lighting and background conditions
Better class balancing
Additional disease classes
Hyperparameter optimization
Extended validation
Confusion-matrix analysis
Cross-dataset testing
Real-world field testing
Continuous retraining using newly collected images
Note

The trained models are part of a prototype AI-based agricultural decision-support system.

Predictions can be affected by image quality, environmental conditions, crop varieties, disease stages and diseases that were not represented in the training data.

For practical crop management, AI predictions should be verified using appropriate agricultural expertise.
