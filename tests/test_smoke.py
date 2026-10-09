"""Bounded smoke checks: model shapes, loss, graph, and sample-data pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcdcnet import PCDCNet, haversine_matrix, prediction_loss


def test_forward_shapes_and_backward():
    torch.manual_seed(0)
    model = PCDCNet(num_pollutants=2, num_covariates=8, hidden_size=16)
    batch, history, horizon, stations = 2, 6, 9, 5
    adjacency = torch.eye(stations)
    output = model(
        torch.randn(batch, history, stations, 2),
        torch.randn(batch, history, stations, 8),
        torch.randn(batch, horizon, stations, 8),
        adjacency,
    )
    assert output["prediction"].shape == (batch, horizon, stations, 2)

    loss = prediction_loss(output["prediction"], torch.zeros_like(output["prediction"]))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_haversine_known_distance():
    # Beijing (116.4E, 39.9N) to Tianjin (117.2E, 39.1N) is roughly 110 km.
    d = haversine_matrix(np.array([116.4, 117.2]), np.array([39.9, 39.1]))
    assert 100 < d[0, 1] < 125
    assert d[0, 0] == 0


def test_sample_dataset_roundtrip():
    sample = Path(__file__).resolve().parents[1] / "data/sample/knowair_v2_bthsa_2017-01.nc"
    if not sample.exists():
        pytest.skip("bundled sample not present")
    from pcdcnet import KnowAirV2, build_station_graph

    dataset = KnowAirV2(
        str(sample),
        "train",
        history=24,
        horizon=72,
        splits={"train": (2017, 2017), "val": (2017, 2017), "test": (2017, 2017)},
    )
    assert len(dataset) > 500
    item = dataset[0]
    assert item["pollutant_history"].shape[0] == 24
    assert item["covariate_future"].shape[0] == 72
    graph = build_station_graph(str(sample.parent / "stations_bthsa.csv"), dataset.stations)
    assert graph["adjacency"].shape[0] == len(dataset.stations)
    restored = dataset.denormalize_pollutants(item["pollutant_future"])
    assert torch.isfinite(restored).all()


def test_optional_emission_channels(tmp_path):
    sample = Path(__file__).resolve().parents[1] / "data/sample/knowair_v2_bthsa_2017-01.nc"
    if not sample.exists():
        pytest.skip("bundled sample not present")
    import xarray as xr

    from pcdcnet import KnowAirV2
    from pcdcnet.data import EMISSION_SPECIES

    with xr.open_dataset(sample) as ds:
        rng = np.random.default_rng(0)
        emissions = xr.Dataset(
            {
                name: (("time", "station"), rng.random(ds["PM2.5"].shape, dtype=np.float32))
                for name in EMISSION_SPECIES
            },
            coords={"time": ds["time"], "station": ds["station"]},
        )
    path = tmp_path / "synthetic_emissions.nc"
    emissions.to_netcdf(path)

    dataset = KnowAirV2(
        str(sample),
        "train",
        history=24,
        horizon=72,
        splits={"train": (2017, 2017), "val": (2017, 2017), "test": (2017, 2017)},
        emissions_path=str(path),
    )
    assert dataset.covariates.shape[-1] == 8 + len(EMISSION_SPECIES)
    assert dataset.covariate_names[-1] == "emission_SO2"

    model = PCDCNet(num_pollutants=2, num_covariates=dataset.covariates.shape[-1])
    item = dataset[0]
    output = model(
        item["pollutant_history"][None],
        item["covariate_history"][None],
        item["covariate_future"][None],
        torch.eye(len(dataset.stations)),
    )
    assert output["prediction"].shape[1] == 72
