# Deep Learning-Based Prediction of Marine Heatwaves in the East China Sea

This repository provides the official PyTorch implementation of the **SwinTrans-ConvLSTM** model for marine heatwave (MHW) prediction, as described in our manuscript submitted to *Ocean Science*.

## 1. Overview
Accurate sea surface temperature (SST) and MHW prediction in highly dynamic shelf seas is challenging. Here, we provide the core network architecture and a minimal reproducible example (demo) to demonstrate our hybrid spatiotemporal framework. 

The model couples:
- **Swin Transformer**: For global spatial representation.
- **ConvLSTM**: For local temporal-evolution modeling.
- **Physics-Informed Curriculum Loss**: Integrating air-sea physical constraints (e.g., spatial gradients) and Online Hard Example Mining (OHEM) to capture extreme MHW events effectively.

## 2. Repository Structure
- `model.py`: Contains the class definitions for the SwinTrans-ConvLSTM architecture and the custom loss function.
- `demo.py`: A minimal reproducible script to run a forward and backward pass of the model.
- `requirements.txt`: Environment dependencies.

## 3. Data Note for Reviewers
Please note that the current `demo.py` utilizes **synthetic data tensors** (`torch.randn`) to demonstrate the structural correctness and gradient updates of the network without requiring the download of massive netCDF climate files. 

The extensive earth science data processing libraries (e.g., `xarray`, `netCDF4`, `PyWavelets`) listed in `requirements.txt` are part of our full preprocessing pipeline. The complete data extraction scripts (from raw netCDF to model inputs) will be made fully open-source upon the acceptance of the manuscript.

### Expected Input Shape
If you wish to test the model with your own tensor slices, ensure the input shape matches:
`[Batch_Size, Spatial_H, Spatial_W, Time_Window, Features]`

## 4. Quick Start (Demo)
To verify the model architecture, please set up the environment and execute the following commands:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo script
python demo.py