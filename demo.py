# -*- coding: utf-8 -*-
"""
Minimal Reproducible Example (Demo) for SwinTrans-ConvLSTM
This script demonstrates the forward pass and loss calculation using synthetic data.
"""
import torch
import torch.optim as optim
from model import SwinConvLSTMModel, PhysicsCurriculumLoss


def run_demo():
    print("=" * 60)
    print("🚀 Starting Demo: SwinTrans-ConvLSTM for MHW Prediction")
    print("=" * 60)

    # 1. Define Model Hyperparameters (matching the manuscript)
    BATCH_SIZE = 2  # You can change this to 4, 8, etc., to test memory usage
    SPATIAL_H, SPATIAL_W = 100, 100
    WINDOW_SIZE = 14  # Input past 14 days
    PRED_STEPS = 7  # Predict next 7 days
    INPUT_FEATURES = 23  # Active physical features (SST, u10, v10, Delta T, etc.)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # 2. Instantiate Model and Loss Function
    print("[*] Initializing the SwinTrans-ConvLSTM model...")
    model = SwinConvLSTMModel(
        input_dim=INPUT_FEATURES,
        spatial_dims=(SPATIAL_H, SPATIAL_W),
        pred_steps=PRED_STEPS
    ).to(device)

    loss_fn = PhysicsCurriculumLoss(start_ohem_epoch=20, ohem_ratio=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)

    # 3. Generate Synthetic Data (Mocking the real NetCDF processing)
    print("[*] Generating synthetic data tensors for demonstration...")
    # Input shape expected by model: [Batch, H, W, Time, Features]
    dummy_inputs = torch.randn(BATCH_SIZE, SPATIAL_H, SPATIAL_W, WINDOW_SIZE, INPUT_FEATURES).to(device)

    # Target shape: [Batch, Pred_Steps, H, W]
    dummy_targets = torch.randn(BATCH_SIZE, PRED_STEPS, SPATIAL_H, SPATIAL_W).to(device)

    # [FIXED]: Dynamic generation of seasonal weights matching the BATCH_SIZE
    # 1.0 represents Summer, 0.0 represents other seasons
    is_summer = torch.randint(0, 2, (BATCH_SIZE,), dtype=torch.float32)
    seasonal_weights = (1.0 + is_summer * 1.0).view(-1, 1, 1, 1).to(device)

    # 4. Perform a Training Step
    print("[*] Performing Forward Pass...")
    model.train()
    optimizer.zero_grad()

    # Outputs from model: [Batch, H, W, Pred_Steps]
    outputs = model(dummy_inputs)
    # Permute to match targets: [Batch, Pred_Steps, H, W]
    outputs = outputs.permute(0, 3, 1, 2)

    print(f"    - Input shape : {dummy_inputs.shape}")
    print(f"    - Output shape: {outputs.shape}")

    print("[*] Calculating Custom Physics-Informed Loss (Epoch 25: OHEM Active)...")
    loss = loss_fn(outputs, dummy_targets, seasonal_weights, current_epoch=25)
    print(f"    - Loss value  : {loss.item():.4f}")

    print("[*] Performing Backward Pass (Gradient update)...")
    loss.backward()
    optimizer.step()

    print("\n✅ SUCCESS: The model architecture and training step are fully functional!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()