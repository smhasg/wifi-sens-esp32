# ESP32 WiFi Sensing for Human Presence Detection

A complete end-to-end WiFi sensing project that uses **ESP32-S3 Channel State Information (CSI)** to detect human presence through machine learning.

The project includes:

- ESP32-S3 transmitter and receiver
- CSI data collection
- Automatic CSV logging
- Data preprocessing
- Feature engineering
- Data visualization
- Machine Learning models
- Real-time analysis

---

# Project Overview

WiFi signals change when a person moves or stands inside a room. Instead of using cameras, this project analyzes **Channel State Information (CSI)** extracted from ESP32-S3 devices to recognize environmental changes.

The collected CSI data is processed and transformed into machine learning features for human presence detection.

---

# Features

- ESP32-S3 CSI Receiver
- CSI Data Logger
- Automatic CSV Storage
- Data Cleaning
- Feature Scaling
- PCA
- t-SNE
- UMAP
- Feature Importance Analysis
- Confusion Matrix
- XGBoost Classifier
- Random Forest Classifier


---

# Machine Learning Pipeline

1. Collect CSI packets
2. Store data as CSV
3. Remove invalid samples
4. Normalize features
5. Visualize CSI patterns
6. Train ML models
7. Evaluate performance

---

# Visualizations

## Dataset Distribution

## Mean CSI Heatmap

## PCA Projection

## t-SNE Projection

## UMAP Projection

## Correlation Matrix

## Feature Importance

## Confusion Matrix

## CSI Signal Example

## CSI Waterfall Visualization

# Technologies

- ESP32-S3
- PlatformIO
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- UMAP
- PCA
- t-SNE

---

# Future Work

- Real-time dashboard
- Multi-person detection
- Human activity recognition
- Gesture recognition
- Deep learning models
- Streamlit visualization

---
