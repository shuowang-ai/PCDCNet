#!/usr/bin/env python
"""Produce one 72-hour forecast from a trained checkpoint.

Picks the first forecast valid time (default: the latest complete forecast
window), runs the model once, and writes a tidy CSV with columns
``valid_time, station, PM2.5, O3`` in physical units.

    python scripts/predict.py \
        --checkpoint checkpoints/bthsa/model.pt \
        --dataset data/knowair_v2/dataset_bthsa.nc \
        --stations data/knowair_v2/stations_bthsa.csv \
        --init-time 2023-06-01T00:00 \
        --output forecast.csv
"""

from __future__ import annotations

import argparse

import _path  # noqa: F401
import numpy as np
import pandas as pd
import torch
import xarray as xr

from pcdcnet import Normalization, PCDCNet, build_station_graph
from pcdcnet.data import METEOROLOGY, POLLUTANTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stations", required=True)
    parser.add_argument("--emissions", help="optional emission NetCDF, as used by train.py")
    parser.add_argument(
        "--init-time", help="ISO valid time of the first forecast lead (one hour after issuance)"
    )
    parser.add_argument("--output", default="forecast.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    history, horizon = config["history"], config["horizon"]
    normalization = Normalization.from_state_dict(state["normalization"])
    covariate_names = config.get("covariates", list(METEOROLOGY))

    with xr.open_dataset(args.dataset) as ds:
        stations = [str(s) for s in ds["station"].values]
        if stations != state["stations"]:
            raise RuntimeError("station order of the dataset does not match the checkpoint")
        times = pd.to_datetime(ds["time"].values)
        if args.init_time:
            start = times.get_loc(pd.Timestamp(args.init_time))
        else:
            start = len(times) - horizon
        if start < history or start + horizon > len(times):
            raise ValueError(
                f"initialization needs {history} history hours and {horizon} future "
                f"covariate hours inside the file ({times[0]} .. {times[-1]})"
            )
        pollutants = np.stack([ds[v].values[start - history : start] for v in POLLUTANTS], axis=-1)
        meteorology_names = [v for v in covariate_names if not v.startswith("emission_")]
        missing = [v for v in meteorology_names if v not in ds.data_vars]
        if missing:
            raise ValueError(f"dataset lacks checkpoint covariates {missing}")
        channels = {
            name: ds[name].values[start - history : start + horizon] for name in meteorology_names
        }
        valid_times = times[start : start + horizon]
        window_times = times[start - history : start + horizon]

    emission_names = [v for v in covariate_names if v.startswith("emission_")]
    if emission_names:
        if not args.emissions:
            raise ValueError("checkpoint requires the matching --emissions file")
        with xr.open_dataset(args.emissions) as emis:
            if [str(s) for s in emis["station"].values] != stations:
                raise ValueError("emission file station order differs from the dataset")
            if not np.array_equal(emis["time"].values, times.values):
                raise ValueError("emission file time axis differs from the dataset")
            for name in emission_names:
                variable = name.removeprefix("emission_")
                if variable not in emis.data_vars:
                    raise ValueError(f"emission file lacks checkpoint covariate {name}")
                channels[name] = emis[variable].values[start - history : start + horizon]
    if not covariate_names or len(set(covariate_names)) != len(covariate_names):
        raise ValueError("checkpoint covariate names must be nonempty and unique")
    covariates = np.stack([channels[name] for name in covariate_names], axis=-1)
    if len(normalization.covariate_mean) != len(covariate_names):
        raise ValueError("checkpoint covariate names and normalization dimensions differ")
    if not (np.diff(window_times.values) == np.timedelta64(1, "h")).all():
        raise ValueError("forecast window must contain consecutive hourly timestamps")
    if not np.isfinite(pollutants).all() or not np.isfinite(covariates).all():
        raise ValueError("forecast history and covariates must be finite")

    pol_std = np.where(normalization.pollutant_std < 1e-6, 1.0, normalization.pollutant_std)
    cov_std = np.where(normalization.covariate_std < 1e-6, 1.0, normalization.covariate_std)
    pollutants = (pollutants - normalization.pollutant_mean) / pol_std
    covariates = (covariates - normalization.covariate_mean) / cov_std

    graph = build_station_graph(args.stations, stations)
    model = PCDCNet(
        num_pollutants=len(POLLUTANTS),
        num_covariates=len(normalization.covariate_mean),
        hidden_size=config["hidden_size"],
    ).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()

    as_tensor = lambda a: torch.from_numpy(a.astype(np.float32))[None].to(args.device)
    with torch.no_grad():
        output = model(
            as_tensor(pollutants[:history]),
            as_tensor(covariates[:history]),
            as_tensor(covariates[history:]),
            graph["adjacency"].to(args.device),
        )
    forecast = (
        output["prediction"][0].cpu().numpy() * normalization.pollutant_std
        + normalization.pollutant_mean
    )

    rows = pd.DataFrame(
        {
            "valid_time": np.repeat(valid_times, len(stations)),
            "station": np.tile(stations, horizon),
            **{
                name: forecast[:, :, channel].reshape(-1) for channel, name in enumerate(POLLUTANTS)
            },
        }
    )
    rows.to_csv(args.output, index=False)
    print(
        f"first forecast valid time {valid_times[0]}, "
        f"{horizon} hours x {len(stations)} stations -> {args.output}"
    )
    summary = rows.groupby("valid_time")[POLLUTANTS].mean().iloc[[0, horizon // 2 - 1, -1]]
    print(summary.round(1).to_string())


if __name__ == "__main__":
    main()
