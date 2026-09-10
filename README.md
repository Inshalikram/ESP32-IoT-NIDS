# ESP32-IoT-NIDS
# ESP32-IoT-NIDS

An ESP32-S3-based IoT Network Intrusion Detection System (NIDS) that detects network attacks in real time using on-device machine learning.

## Overview

This project combines embedded firmware, machine learning, and a web dashboard to monitor WiFi traffic and detect malicious activity directly on an ESP32-S3 microcontroller no cloud dependency required for inference.

## Key Features

- **On-device ML inference** using TensorFlow Lite (TFLite Micro) running directly on the ESP32-S3
- **8 attack class detection** including ARP spoofing, MQTT-based DoS/DDoS, TCP/IP DoS/DDoS, and reconnaissance (port scan, ping sweep, vulnerability scan)
- **99.43% detection accuracy** achieved through feature engineering and model optimization (LightGBM + TFLite)
- **Promiscuous mode packet sniffing** for passive WiFi traffic capture
- **FreeRTOS dual-core architecture** for parallel packet capture and inference
- **Real-time alerting** via MQTT (HiveMQ Cloud)
- **Live dashboard** built with Flask, using Server-Sent Events (SSE) for real-time updates
- **Fully local deployment** on a static-IP setup (no reliance on third-party hosting)

## Tech Stack

| Component | Technology |
|---|---|
| Firmware | ESP32-S3, Arduino/C++, FreeRTOS |
| ML Inference | TensorFlow Lite Micro |
| Model Training | Python, LightGBM |
| Backend | Flask |
| Alerting | MQTT (HiveMQ Cloud) |
| Real-time updates | Server-Sent Events (SSE) |

## Project Structure

```
├── data/                    # Training/test datasets (not included — see Dataset section)
├── models/                  # Trained models (.pkl, .tflite, .h, .keras)
├── preprocessing.py         # Data preprocessing
├── feature eng.py           # Feature engineering
├── prepare_dataset.py       # Dataset preparation
├── merge_data.py            # Merge train/test data
├── utils_features.py        # Feature utility functions
├── check_dataset.py         # Dataset validation
├── train_model.py           # Model training
├── train_tiny _ml.py        # TinyML model training
├── cross_validation.py      # Cross-validation
├── model _evaluation.py     # Model evaluation
├── tflite_to_header.py      # Convert TFLite model to C header for firmware
├── scaler.pkl               # Feature scaler
└── requirements.txt         # Python dependencies
```

## Dataset

The training and test datasets are not included in this repository due to size. [Add your dataset source/link here, e.g. Google Drive link or dataset name/citation.]

## Setup

1. Clone the repository
   ```
   git clone https://github.com/Inshalikram/ESP32-IoT-NIDS.git
   cd ESP32-IoT-NIDS
   ```
2. Install Python dependencies
   ```
   pip install -r requirements.txt
   ```
3. [Add firmware flashing instructions for the ESP32-S3, if applicable]
4. [Add dashboard/server run instructions, e.g. `python models/server.py`]

## Model Performance

- **Accuracy:** 99.43%
- **Attack classes detected:** 8 (WiFi/network-based attacks)

## Author

Inshalikram — Final Year Project
