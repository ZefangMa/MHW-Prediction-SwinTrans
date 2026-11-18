"""Run an SST forecasting example using only synthetic daily fields.

The grid, feature subset, reference SST, and single training update are for
illustration. This script does not reproduce the manuscript experiments.
"""

import argparse
from datetime import date, timedelta

import numpy as np
import torch

from model import CurriculumGradientLoss, SwinConvLSTMModel, predict_sst


HISTORY_DAYS = 14
FORECAST_DAYS = 7
# Each half-open interval refers to target dates, not initialization dates.
PERIODS = {"train": (0, 55), "validation": (55, 80), "test": (80, 105)}
FEATURES = (
    "sst_anomaly", "sst_backward_difference", "u10", "v10", "t2m",
    "air_sea_temperature_difference", "u10_trailing_mean3", "v10_trailing_mean3",
    "air_sea_temperature_difference_trailing_mean3",
)


def synthetic_fields():
    """Return [day, latitude, longitude, field] and a fixed ocean mask."""
    rng = np.random.default_rng(7)
    day = np.arange(105, dtype=np.float32)[:, None, None]
    y, x = np.mgrid[:20, :20]
    ocean = (x > 3 + 1.2 * np.sin(y / 3)) & ((y - 13)**2 + (x - 13)**2 > 4)
    sst = 18 + 0.045 * day + 0.12 * y - 0.06 * x + np.sin(day / 9)
    u10 = 2 + np.cos(day / 5) + 0.08 * x + 0.02 * y
    v10 = np.sin(day / 7) - 0.03 * y + 0.02 * x
    t2m = sst - 0.7 + 0.6 * np.cos(day / 6 + x / 7)
    fields = np.stack((sst, u10, v10, t2m), axis=-1).astype(np.float32)
    fields += rng.normal(0, 0.12, fields.shape).astype(np.float32)
    fields[:, ~ocean] = np.nan
    fields[20:22, 8, 11, 0] = np.nan  # missing historical SST
    fields[[50, 80], 8, 11, 0] = np.nan  # missing training and test targets
    return fields, ocean


def fit_input_fallback_and_reference(fields, ocean, train_stop):
    """Use training dates only for input fallback and a toy SST reference."""
    train = fields[:train_stop]
    fallback = np.nanmean(train[:, ocean], axis=(0, 1))
    reference = np.zeros(ocean.shape, dtype=np.float32)
    reference[ocean] = np.nanmean(train[..., 0][:, ocean], axis=0)
    return fallback, reference


def trailing_mean3(field):
    return np.stack([field[max(0, day - 2):day + 1].mean(axis=0)
                     for day in range(len(field))])


def historical_features(prefix, fallback, reference):
    """Build features from a prefix ending on the forecast initialization date."""
    filled = np.empty_like(prefix)
    last = np.broadcast_to(fallback, prefix.shape[1:])
    for day, values in enumerate(prefix):
        last = np.where(np.isfinite(values), values, last)
        filled[day] = last
    sst, u10, v10, t2m = np.moveaxis(filled, -1, 0)
    delta_t = sst - t2m
    return np.stack((
        sst - reference, np.diff(sst, axis=0, prepend=sst[:1]),
        u10, v10, t2m, delta_t, trailing_mean3(u10),
        trailing_mean3(v10), trailing_mean3(delta_t),
    ), axis=-1)


def valid_origins(start, stop):
    """Keep only windows with all seven targets in one split."""
    return range(max(HISTORY_DAYS - 1, start - 1), stop - FORECAST_DAYS)


def run_demo(device, flip_average=False):
    fields, ocean = synthetic_fields()
    dates = [date(2000, 5, 1) + timedelta(days=day) for day in range(len(fields))]
    train_stop = PERIODS["train"][1]
    fallback, reference = fit_input_fallback_and_reference(fields, ocean, train_stop)
    train_features = historical_features(fields[:train_stop], fallback, reference)[:, ocean]
    input_mean = train_features.mean(axis=(0, 1))
    input_std = np.maximum(train_features.std(axis=(0, 1)), 1e-6)
    train_anomalies = (fields[:train_stop, ..., 0] - reference)[:, ocean]
    target_mean = np.nanmean(train_anomalies)
    target_std = np.maximum(np.nanstd(train_anomalies), np.float32(1e-6))

    def model_input(origin, source):
        # No data after the initialization date enters historical_features.
        prefix = source[:origin + 1]
        window = historical_features(prefix, fallback, reference)[-HISTORY_DAYS:]
        window = ((window - input_mean) / input_std).transpose(1, 2, 0, 3)
        window[~ocean] = 0
        return window.copy()

    def sample(origin):
        days = slice(origin + 1, origin + FORECAST_DAYS + 1)
        raw_sst = fields[days, ..., 0].copy()
        target = (raw_sst - reference - target_mean) / target_std
        weights = np.array([2.0 if d.month in (6, 7, 8) else 1.0
                            for d in dates[days]], dtype=np.float32)
        weights = torch.from_numpy(weights[None, :, None, None]).to(device)
        return target, raw_sst, weights

    origins = {name: valid_origins(*period) for name, period in PERIODS.items()}
    for name, (start, stop) in PERIODS.items():
        first, last = origins[name][0], origins[name][-1]
        assert start <= first + 1 <= last + FORECAST_DAYS < stop
        print(f"{name:>10}: targets {dates[start]} to {dates[stop - 1]}, "
              f"{len(origins[name])} eligible forecast origins")

    test_origin = origins["test"][0]
    changed = fields.copy()
    changed[test_origin + 1:] += 50
    np.testing.assert_array_equal(model_input(test_origin, fields), model_input(test_origin, changed))
    print("Synthetic causality check passed: changing future fields leaves historical inputs unchanged.")

    def torch_input(origin):
        return torch.from_numpy(model_input(origin, fields)[None]).to(device)

    torch.manual_seed(7)
    mask = torch.from_numpy(ocean).to(device)
    model = SwinConvLSTMModel(len(FEATURES), ocean.shape, FORECAST_DAYS).to(device)
    criterion = CurriculumGradientLoss(mask).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_origin = origins["train"][-1]
    inputs = torch_input(train_origin)
    target, _, weights = sample(train_origin)
    target = torch.from_numpy(target[None]).to(device)
    optimizer.zero_grad()
    prediction = model(inputs).permute(0, 3, 1, 2)
    loss = criterion(prediction, target, weights, epoch=0)
    loss.backward()
    optimizer.step()
    print(f"Training example: input {tuple(inputs.shape)}, "
          f"standardized forecast {tuple(prediction.shape)}, loss {loss.item():.4f}")

    model.eval()
    for name in ("validation", "test"):
        origin = origins[name][0]
        _, raw_sst, _ = sample(origin)
        forecast = predict_sst(model, torch_input(origin), reference, target_mean, target_std,
                               mask, flip_average=flip_average)[0].cpu().numpy()
        valid = ocean[None] & np.isfinite(raw_sst)
        assert np.isnan(forecast[:, ~ocean]).all() and np.isfinite(forecast[:, ocean]).all()
        mae = np.mean(np.abs(forecast[valid] - raw_sst[valid]))
        print(f"{name:>10} example: initialized {dates[origin]}, "
              f"targets {dates[origin + 1]} to {dates[origin + FORECAST_DAYS]}; "
              f"synthetic MAE {mae:.3f} °C over {valid.sum()} ocean grid-day pairs")
    print("Synthetic demonstration only; values are not manuscript results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--flip-average", action="store_true",
                        help="Also use longitude flip averaging for inference.")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 2))
    elif not torch.cuda.is_available():
        parser.error("CUDA is not available on this machine.")
    run_demo(torch.device(args.device), flip_average=args.flip_average)
