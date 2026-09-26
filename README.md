# Computational Optimization of Object Detection Models

## Project Overview

This project focuses on reducing the computational requirements of an object detection model while maintaining acceptable detection performance.

The classroom monitoring system is used as a case study to evaluate the optimization techniques.

## Objectives

- Reduce model computational complexity
- Reduce the number of parameters
- Reduce MACs/GFLOPs
- Improve inference speed and FPS
- Reduce model size where possible
- Maintain acceptable precision, recall and mAP
- Compare the original and optimized models

## Optimization Techniques

The project investigates techniques such as:

- Input resolution optimization
- Structural pruning
- Fine-tuning after pruning
- Model compression
- Quantization

## Model

The project uses a custom YOLO-based object detection model.

Baseline model:

- Parameters: approximately 25.86M
- GFLOPs: approximately 78.7
- Model size: approximately 50 MB

## Case Study

The optimized model is evaluated using a classroom monitoring dataset containing objects relevant to the application.

The classroom is the application case study; the primary research focus is computational model optimization.

## Technology Stack

- Python
- PyTorch
- Ultralytics YOLO
- Torch-Pruning
- OpenCV
- Google Colab
- Streamlit
- GitHub

## Project Structure

```text
smart-classroom-monitor/
│
├── README.md
├── app.py
├── requirements.txt
├── runtime.txt
│
├── notebooks/
│   └── YOLO_Optimization_Master.ipynb
│
├── scripts/
│   ├── baseline.py
│   ├── validation.py
│   ├── speed_test.py
│   └── pruning.py
│
└── results/
