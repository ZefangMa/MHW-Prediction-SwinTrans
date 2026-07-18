# Deep Learning-Based Prediction of Marine Heatwaves in the East China Sea

This repository provides the official PyTorch implementation of the **SwinTrans-ConvLSTM** model for marine heatwave (MHW) prediction, as described in our manuscript submitted to *Ocean Science*.

## 1. Overview
Accurate sea surface temperature (SST) and MHW prediction in highly dynamic shelf seas is challenging. Here, we provide the core network architecture and a minimal reproducible example (demo) to demonstrate our hybrid spatiotemporal framework. 

The model couples:
- **Swin Transformer**: For global spatial representation.
- **ConvLSTM**: For local temporal-evolution modeling.
- **Physics-Informed Constraints**: Integrating air-sea temperature difference and vector wind fields.

## 2. Repository Structure
- `model.py`: Contains the class definitions for the SwinTrans-ConvLSTM architecture.
- `demo.py`: A minimal reproducible script to run a forward pass of the model with synthetic or sample data.
- `requirements.txt`: Environment dependencies.

## 3. Quick Start (Demo)
To verify the model architecture and run the demo script, please set up the environment and execute the following commands:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo script
python demo.py
