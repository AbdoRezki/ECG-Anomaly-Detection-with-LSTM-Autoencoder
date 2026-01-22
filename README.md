# ECG Anomaly Detection with LSTM Autoencoder

End-to-end deep learning system that detects anomalous heartbeats in ECG signals using a PyTorch LSTM autoencoder trained on the ECG5000 dataset, with Flask API deployment for real-time inference.

## Overview

This project uses an autoencoder architecture with LSTM layers to learn normal heartbeat patterns from ECG time series data (140 timesteps per sequence). The model flags anomalies like PVC (Premature Ventricular Contraction) and R-on-T beats by measuring reconstruction loss—abnormal patterns produce high reconstruction errors.

## Features

- **LSTM-based autoencoder**: Captures temporal dependencies in ECG sequences with encoder (128→64 hidden units) and decoder (64→128→1)
- **Anomaly detection**: Reconstruction loss thresholding on validation data (~26 MAE) achieves ~98% accuracy on normal/anomaly classification
- **Flask API**: REST endpoint for real-time anomaly scoring of new ECG sequences
- **Visualization**: Training loss curves and anomaly score distributions

## Dataset

**ECG5000** from UCI Time Series Classification Archive:
- 5,000 heartbeat sequences (140 timesteps each, 20Hz sampling)
- Class 1: Normal beats (~90%)
- Classes 2-5: Anomalous patterns (PVC, R-on-T, etc.)
- Download: http://www.timeseriesclassification.com/

## Project Structure

```text
ecg-anomaly-detection-pytorch/
├── data/
│ ├── ECG5000_TRAIN.arff
│ └── ECG5000_TEST.arff
├── src/
│ ├── init.py
│ ├── data_prep.py # Data loading & preprocessing
│ └── model.py # LSTM autoencoder architecture
├── train.py # Training script
├── predict_eval.py # Evaluation & threshold tuning
├── app.py # Flask API
├── requirements.txt
├── model.pth # Trained model weights
├── loss.png # Training history plot
└── README.md

```

