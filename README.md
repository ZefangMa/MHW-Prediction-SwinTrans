# Deep Learning-Based Prediction of Marine Heatwaves in the East China Sea

This repository provides the official PyTorch implementation of the **SwinTrans-ConvLSTM** model for marine heatwave (MHW) prediction.

## 1. Overview
Accurate sea surface temperature (SST) and MHW prediction in highly dynamic shelf seas is challenging. This repository contains the core network architecture and a minimal reproducible example to demonstrate our hybrid spatiotemporal framework. 

The model couples:
- **Swin Transformer**: For global spatial representation.
- **ConvLSTM**: For local temporal-evolution modeling.
- **Physics-Informed Curriculum Loss**: Integrating air-sea physical constraints (e.g., spatial gradients) and Online Hard Example Mining (OHEM) to capture extreme MHW events effectively.

## 2. Repository Structure
- `model.py`: Contains the class definitions for the SwinTrans-ConvLSTM architecture and the custom loss function.
- `demo.py`: A minimal reproducible script to run a forward and backward pass of the model.
- `requirements.txt`: Environment dependencies for the model and data processing pipelines.

## 3. Data Description & Usage
To facilitate quick verification of the model architecture without requiring the download of massive gigabyte-scale netCDF climate files, the current `demo.py` utilizes **synthetic data tensors** (`torch.randn`). This allows researchers and users to test the structural correctness and gradient updates instantly on local machines.

### Expected Input Shape
If you wish to test the model with your own regional ocean dataset, please ensure the input tensor shape matches:
`[Batch_Size, Spatial_H, Spatial_W, Time_Window, Features]`

*(Note: The full data processing scripts, from raw netCDF to model inputs, will be fully open-sourced following the publication of our related research.)*

## 4. Quick Start
To verify the model architecture, set up the environment and execute the following commands:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo script
python demo.py
```

## 5. Contact
For any questions regarding the code or the methodology, please feel free to open an Issue in this repository, or contact via email at 2024110101007@zjou.edu.cn.
