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
