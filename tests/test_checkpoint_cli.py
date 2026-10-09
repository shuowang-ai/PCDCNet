"""Checkpoint input contracts across training data, evaluation and prediction."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from pcdcnet import KnowAirV2, PCDCNet, build_station_graph, evaluate
from pcdcnet.data import METEOROLOGY, POLLUTANTS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("with_emissions", [False, True])
def test_checkpoint_channels_through_both_clis(tmp_path, with_emissions):
    # Distinct channel scales reveal wrong selections or channel ordering.
    rng = np.random.default_rng(7)
    times = np.concatenate(
        [pd.date_range(f"{year}-01-01", periods=8, freq="h").values for year in (2017, 2022)]
    )
    stations = ["a", "b"]
    shape = (len(times), len(stations))
    data = xr.Dataset(
        {
            name: (("time", "station"), (i + 1) * rng.uniform(1, 5, shape).astype("float32"))
            for i, name in enumerate(POLLUTANTS + METEOROLOGY)
        },
        coords={"time": times, "station": stations},
    )
    dataset_path = tmp_path / "dataset.nc"
    data.to_netcdf(dataset_path)
    stations_path = tmp_path / "stations.csv"
    pd.DataFrame({"station_id": stations, "lon": [116.4, 116.5], "lat": [39.9, 39.8]}).to_csv(
        stations_path, index=False
    )
    names = ["v100", "t2m", "d2m", "sp", "tp", "blh", "u100"]
    emissions_path = None
    if with_emissions:
        emissions_path = tmp_path / "emissions.nc"
        xr.Dataset(
            {"NOx": (("time", "station"), rng.uniform(10, 20, shape).astype("float32"))},
            coords=data.coords,
        ).to_netcdf(emissions_path)
        names += ["emission_NOx"]
    train = KnowAirV2(
        str(dataset_path),
        history=2,
        horizon=3,
        covariate_names=names,
        emissions_path=emissions_path,
    )
    # This expectation comes from raw named variables, not the selection code.
    for c, name in enumerate(names[:7]):
        raw = data[name].values[:8]
        np.testing.assert_allclose(train.normalization.covariate_mean[c], raw.mean(), rtol=1e-6)
        np.testing.assert_allclose(
            train.covariates[..., c], (raw - raw.mean()) / raw.std(), rtol=1e-5, atol=1e-6
        )
    test = KnowAirV2(
        str(dataset_path),
        "test",
        history=2,
        horizon=3,
        normalization=train.normalization,
        covariate_names=names,
        emissions_path=emissions_path,
    )
    torch.manual_seed(5)
    model = PCDCNet(2, len(names), hidden_size=8)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "stations": stations,
            "normalization": train.normalization.state_dict(),
            "config": {"history": 2, "horizon": 3, "hidden_size": 8, "covariates": names},
        },
        checkpoint,
    )
    graph = build_station_graph(str(stations_path), stations)
    expected = evaluate(model, test, graph["adjacency"], device="cpu", num_workers=0)
    common = [
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        str(dataset_path),
        "--stations",
        str(stations_path),
        "--device",
        "cpu",
    ]
    if emissions_path:
        common += ["--emissions", str(emissions_path)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if not with_emissions:
        # Two minibatches with accumulation=4 used to perform no optimizer step.
        run_dir = tmp_path / "trained"
        train_args = [
            sys.executable,
            str(ROOT / "scripts/train.py"),
            "--demo",
            "--max-epochs",
            "1",
            "--dataset",
            str(dataset_path),
            "--stations",
            str(stations_path),
            "--history",
            "2",
            "--horizon",
            "3",
            "--hidden-size",
            "8",
            "--batch-size",
            "1",
            "--accumulate",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]
        subprocess.run(
            [*train_args, "--output", str(run_dir)],
            check=True,
            env=env,
            capture_output=True,
        )
        trained = torch.load(run_dir / "model.pt", weights_only=True)
        torch.manual_seed(42)
        initial = PCDCNet(2, 8, hidden_size=8)
        assert not torch.equal(trained["model"]["embed.weight"], initial.embed.weight)
    metrics = tmp_path / "metrics.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/evaluate.py"), *common, "--output", str(metrics)],
        check=True,
        env=env,
        capture_output=True,
    )
    actual = json.loads(metrics.read_text())
    assert actual["num_windows"] == 4
    assert actual["MAE"] == expected["MAE"]
    assert actual["RMSE"] == expected["RMSE"]

    # Future target observations are deliberately missing: inference must use history only.
    for name in POLLUTANTS:
        data[name].values[10:13] = np.nan
    data.to_netcdf(dataset_path)
    forecast = tmp_path / "forecast.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/predict.py"),
            *common,
            "--init-time",
            "2022-01-01T02:00",
            "--output",
            str(forecast),
        ],
        check=True,
        env=env,
        capture_output=True,
    )
    rows = pd.read_csv(forecast)
    assert len(rows) == 6
    assert rows.station.tolist() == stations * 3
    assert pd.to_datetime(rows.valid_time).min() == pd.Timestamp("2022-01-01T02:00")
    sample = test[0]
    with torch.no_grad():
        prediction = model(
            sample["pollutant_history"][None],
            sample["covariate_history"][None],
            sample["covariate_future"][None],
            graph["adjacency"],
        )["prediction"]
    expected_forecast = test.denormalize_pollutants(prediction)[0].numpy().reshape(-1, 2)
    np.testing.assert_allclose(rows[POLLUTANTS].values, expected_forecast, rtol=1e-5, atol=1e-5)
