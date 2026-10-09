"""Training statistics remain accurate across large arrays and channel layouts."""

import numpy as np
import pandas as pd
import xarray as xr

from pcdcnet import KnowAirV2
from pcdcnet.data import METEOROLOGY, POLLUTANTS


def test_large_training_statistics_match_analytic_values(tmp_path):
    # 128,000 training values expose float32 accumulation drift at pressure scale.
    # Each channel has two equally frequent, exactly known float32 endpoints.
    train_hours, test_hours, stations = 1000, 8, 128
    times = np.concatenate(
        [
            pd.date_range("2017-01-01", periods=train_hours, freq="h").values,
            pd.date_range("2022-01-01", periods=test_hours, freq="h").values,
        ]
    )
    offsets = [60, 70, 280, 275, 100000, 0.001, 500, 150, 2, 3]
    amplitudes = [10, 20, 4, 3, 32, 0.0002, 50, 30, 1, 2]
    variables = {}
    expected_mean, expected_std = [], []
    for name, offset, amplitude in zip(POLLUTANTS + METEOROLOGY, offsets, amplitudes):
        low, high = np.float32(offset - amplitude), np.float32(offset + amplitude)
        values = np.empty((train_hours + test_hours, stations), dtype=np.float32)
        values[:train_hours:2] = low
        values[1:train_hours:2] = high
        # A conspicuously different held-out distribution must not affect fitting.
        values[train_hours:] = np.float32(offset + 1000000)
        variables[name] = (("time", "station"), values)
        expected_mean.append((float(low) + float(high)) / 2)
        expected_std.append((float(high) - float(low)) / 2)
    path = tmp_path / "large.nc"
    xr.Dataset(
        variables,
        coords={"time": times, "station": [f"s{i}" for i in range(stations)]},
    ).to_netcdf(path)

    default = KnowAirV2(str(path), history=2, horizon=2)
    explicit = KnowAirV2(str(path), history=2, horizon=2, covariate_names=list(METEOROLOGY))
    expected_mean = np.asarray(expected_mean, dtype=np.float32)
    expected_std = np.asarray(expected_std, dtype=np.float32)
    for dataset in (default, explicit):
        norm = dataset.normalization
        actual_mean = np.concatenate([norm.pollutant_mean, norm.covariate_mean])
        actual_std = np.concatenate([norm.pollutant_std, norm.covariate_std])
        np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(actual_std, expected_std, rtol=1e-6, atol=1e-9)
        assert actual_mean.dtype == actual_std.dtype == np.float32

    # Requesting the already-default order must not change the fitted statistics.
    for field in vars(default.normalization):
        np.testing.assert_array_equal(
            getattr(default.normalization, field), getattr(explicit.normalization, field)
        )
    np.testing.assert_array_equal(default.covariates, explicit.covariates)
