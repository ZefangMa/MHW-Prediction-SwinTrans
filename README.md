# SwinTrans-ConvLSTM for SST Forecasting in the East China Sea

SwinTrans-ConvLSTM combines a shifted-window Transformer for spatial feature extraction with ConvLSTM for temporal modeling. It was developed to forecast daily sea surface temperature (SST) in the East China Sea and support the evaluation of marine heatwave (MHW) forecasts.

## What is available here

This repository currently contains a PyTorch model and a demonstration that uses synthetic inputs and targets. The demo illustrates one training step, including a forward pass, loss calculation, and parameter update. It does not use observed SST or produce the forecasts, figures, or tables reported in the manuscript.

| File | Purpose |
| --- | --- |
| [`model.py`](model.py) | Network architecture and loss used in the demonstration |
| [`demo.py`](demo.py) | Training step on synthetic tensors |
| [`requirements.txt`](requirements.txt) | Python dependencies listed for this repository |

## Run the demonstration

```bash
python -m pip install -r requirements.txt
python demo.py
```

The demo input is ordered as `[batch, latitude, longitude, input day, feature]`. Its feature channels and targets are synthetic; the demo configuration does not document the manuscript's input feature set or training protocol.

## Study workflow

In the manuscript, each forecast uses 14 days of inputs ending on the initialization date and generates daily SST forecasts for the following seven days from that initialization. Predictors include features derived from the air–sea temperature difference and surface wind. ERA5 inputs are restricted to the historical input window. Forecasts are evaluated against daily OSTIA SST fields for 2020–2024. The study assesses continuous SST errors and evaluates MHW event days and event characteristics separately.

The data preparation, training, baseline, and evaluation scripts needed to reproduce those results are not included in the current repository. The synthetic demonstration does not require OSTIA or ERA5 data.

## License and questions

The code is available under the [MIT License](LICENSE). For questions about the released code, please open a [GitHub issue](https://github.com/ZefangMa/MHW-Prediction-SwinTrans/issues).
