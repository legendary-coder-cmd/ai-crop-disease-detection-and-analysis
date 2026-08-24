# 🌾 CropAI – Multi-Crop AI Farm Assist

<p align="center">
  <img src="https://img.shields.io/badge/Project-CropAI-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Arduino-UNO%20Q-blue?style=for-the-badge&logo=arduino">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Python-3.x-yellow?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Git%20LFS-Datasets%20%26%20Models-purple?style=for-the-badge">
</p>

<p align="center">
  <b>🤖 AI-powered crop identification and disease detection system</b>
</p>

<p align="center">
  Built using <b>Arduino UNO Q + Computer Vision + TensorFlow/Keras + Camera + Environmental Sensing</b>
</p>

---

## 🌱 About CropAI

**CropAI – Multi-Crop AI Farm Assist** is an AI-based agricultural monitoring system designed to identify crops and detect crop diseases from leaf images.

The system combines:

📷 Camera-based image capture  
🧠 Deep-learning disease classification  
🌾 Crop-specific AI models  
🌡️ DHT11 environmental sensing  
💻 Arduino UNO Q embedded computing  
🌐 Multilingual diagnosis  
🔋 Portable battery-powered hardware  

The current prototype supports **five crops**:

🌽 Corn  
🌿 Cotton  
🌾 Paddy  
🎋 Sugarcane  
🌾 Wheat  

---

## 🚀 Project Status

### 🟢 Multi-Crop AI Prototype / Submission Version

| Component | Status |
|---|:---:|
| 🌽 Corn Dataset | ✅ |
| 🌿 Cotton Dataset | ✅ |
| 🌾 Paddy Dataset | ✅ |
| 🎋 Sugarcane Dataset | ✅ |
| 🌾 Wheat Dataset | ✅ |
| 🧠 Trained AI Models | ✅ |
| 🏋️ Training Script | ✅ |
| 📷 Camera Inference | ✅ |
| 🤖 Arduino UNO Q | ✅ |
| 🌡️ DHT11 Sensor | ✅ |
| 🔋 12 V Power System | ✅ |
| 🌐 Multilingual Diagnosis | ✅ |
| ✏️ Hardware Schematics | ✅ |
| 📸 Prototype Documentation | ✅ |
| 🔬 AI Result Examples | ✅ |
| 📦 Git LFS | ✅ |

---

# ⭐ Key Features

### 🌾 Multi-Crop AI

CropAI supports:

- 🌽 Corn
- 🌿 Cotton
- 🌾 Paddy
- 🎋 Sugarcane
- 🌾 Wheat

### 🦠 Disease Detection

Each crop uses a dedicated disease-classification model.

### 📷 Camera Detection

The system can analyse images captured using the connected webcam.

### 🖼️ Image Upload

Users can provide leaf images for AI-based analysis.

### 🎯 Confidence-Based Prediction

The system uses prediction confidence to help determine whether the detected result is reliable.

### 🌐 Multilingual Diagnosis

The user can select a language and receive the crop/disease information in the selected language.

### 🌡️ Environmental Monitoring

The DHT11 sensor provides:

- Temperature
- Humidity

### 🤖 Embedded AI

The system is designed around the **Arduino UNO Q** platform.

---

# 🔬 How CropAI Works

```text
                 📷 Camera / Image
                         │
                         ▼
                 🖼️ Image Processing
                         │
                         ▼
                  🌾 Crop Detection
                         │
                         ▼
             🧠 Crop-Specific AI Model
                         │
                         ▼
                🦠 Disease Detection
                         │
                         ▼
                  🎯 Confidence
                         │
                         ▼
              📋 Disease Information
                         │
                         ▼
              🌐 Multilingual Diagnosis