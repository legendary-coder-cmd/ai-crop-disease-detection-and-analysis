# ai-crop-disease-detection-and-analysis
AI-based crop disease detection using Arduino UNO Q, webcam (or) camera, and TensorFlow/Keras models.
# CropAI – Multi-Crop AI Farm Assist

## Camera-Based Crop and Disease Detection

CropAI is a modular multi-crop computer-vision system designed to identify crop type and detect diseases from leaf images captured using an external webcam or uploaded image.

The system validates the input, identifies the most likely crop, routes the image to the corresponding crop-specific trained `.keras` disease model, and produces a disease prediction with confidence and field-assistance information.

## Supported Crops

CropAI supports five crops:

* 🌾 Paddy
* 🌿 Cotton
* 🎋 Sugarcane
* 🌽 Corn
* 🌾 Wheat

Each crop uses its own disease-classification model.

## System Pipeline

```text
Camera / Image Upload
        ↓
Leaf Presence & Image Quality Check
        ↓
Crop Identification
        ↓
Crop-Specific .keras Model
        ↓
Multi-View Prediction
        ↓
Probability Averaging
        ↓
Disease Classification
        ↓
Confidence + Field Information
        ↓
Save / Recapture
```

The implementation uses multiple views of the captured leaf and averages the resulting model probabilities to improve prediction stability.

## Crop Disease Classes

### Paddy

* Bacterial Leaf Blight
* Bacterial Leaf Streak
* Bacterial Panicle Blight
* Blast
* Brown Spot
* Healthy
* Tungro

### Cotton

* Bacterial Blight
* Healthy
* Leaf Curl Virus
* Leaf Redding
* Fusarium Wilt

### Sugarcane

* Bacterial Blight
* Healthy
* Mosaic

### Corn

* Common Rust
* Gray Leaf Spot
* Healthy
* Leaf Blight

### Wheat

* Brown Rust
* Fusarium Head Blight
* Healthy
* Septoria
* Tan Spot

## Input Validation

Before classification, the system checks whether the image contains a usable plant/leaf region.

The image is analyzed using OpenCV-based color information and brightness checks. If a suitable leaf is not detected, the system returns:

```text
NO CLEAR LEAF DETECTED
```

and asks the user to place a leaf clearly inside the frame.

## Crop Identification

The available crop-specific models are evaluated to determine the most likely crop.

A crop is accepted only when its confidence passes the configured threshold. If the system cannot confidently distinguish the crop, it returns:

```text
CROP COULD NOT BE IDENTIFIED
```

This prevents an uncertain crop classification from being passed directly to a disease model.

## Disease Detection

After identifying the crop, the image is passed to that crop's disease model.

The predicted class, confidence, probability distribution and ranked predictions are calculated.

The result displayed to the user includes:

* Crop
* Crop confidence
* Disease
* Disease confidence
* Cause
* Treatment/management guidance
* Environmental conditions

## Model Architecture

The project uses crop-specific TensorFlow/Keras models stored in `.keras` format.

Configured models include:

```text
models/
├── paddy_disease_v5_best.keras
├── cotton_disease_v4_best.keras
├── sugarcane_disease_best.keras
├── corn_disease_best.keras
└── wheat_disease_best.keras
```

The original submission document specifies these as the crop-specific models used by the application.

## Camera Interface

The application supports an external webcam.

Controls:

```text
C → Capture image
U → Upload image
R → Recapture
J → Save result
Q → Quit
```

The camera is configured for a 1280 × 720 capture resolution in the submission implementation.

## Dataset

The repository contains the image datasets used for the five supported crops:

```text
datasets/
├── corn dataset/
├── cotton dataset/
├── paddy dataset/
├── sugarcane dataset/
└── wheat dataset/
```

The datasets contain disease/healthy image classes used for training the crop-specific models.

Large image datasets are stored using **Git LFS**.

## Training

The training workflow is documented separately in:

```text
scripts/training/readme.md
```

The training process produces crop-specific `.keras` models that are subsequently loaded by the multi-crop inference system.

The final inference system does not need to load the complete training dataset when an already-trained `.keras` model is available.

## Hardware Documentation

Hand-drawn hardware schematics and circuit documentation are available in:

```text
schematics/
```

These provide the hardware and wiring reference used during project development.

## Repository Structure

```text
ai-crop-disease-detection-and-analysis/
│
├── datasets/
│   ├── corn dataset/
│   ├── cotton dataset/
│   ├── paddy dataset/
│   ├── sugarcane dataset/
│   └── wheat dataset/
│
├── schematics/
│   ├── 01_battery_supply_handdrawn.jpeg
│   ├── 02_block_diagram_and_circuit.jpeg
│   └── readme.md
│
├── scripts/
│   └── training/
│       └── readme.md
│
├── .gitattributes
└── README.md
```

## Field Assistance

The system provides general disease information and management guidance along with the prediction.

The recommendations are intentionally advisory rather than prescribing a universal pesticide dose. Actual product selection, dosage and spray interval should follow the current locally registered product label and appropriate agricultural guidance.

## Safety and Uncertainty

CropAI is designed to avoid treating an uncertain prediction as a confirmed diagnosis.

The system can reject:

* Images without a clear leaf
* Uncertain crop identification
* Low-confidence predictions

A clearer image can then be captured or uploaded again.

## Future Improvements

The current architecture uses individual disease models for each crop.

A stronger production architecture could introduce a dedicated five-class crop classifier:

```text
Image
  ↓
Crop Classifier
  ↓
Paddy / Cotton / Sugarcane / Corn / Wheat
  ↓
Crop-Specific Disease Model
  ↓
Disease Prediction
```

This keeps the disease models completely crop-specific while making crop routing a dedicated classification stage.

## Project Status

**Status: Multi-Crop AI Prototype / Submission Version**

The repository contains:

* Multi-crop image datasets
* Crop-specific disease-classification workflow
* TensorFlow/Keras model support
* Camera-based inference
* Image-upload inference
* Multi-view prediction
* Confidence-based crop routing
* Disease information
* Hand-drawn hardware schematics
* Training documentation

## Disclaimer

CropAI is a prototype intended for project demonstration and decision-support purposes. Disease predictions should be visually verified and, where appropriate, confirmed using qualified agricultural expertise.

Chemical recommendations must follow current local agricultural guidance and registered product labels.
