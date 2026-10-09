"""Station graph construction from monitoring-site coordinates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

EARTH_RADIUS_KM = 6371.0


def haversine_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances in kilometres."""
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_station_graph(
    stations_csv: str,
    station_order: list[str],
    threshold_km: float = 200.0,
) -> dict[str, torch.Tensor]:
    """Build the proximity graph used by PCDCNet.

    Two stations are connected when their great-circle distance is at most
    ``threshold_km``. The returned adjacency is the symmetrically normalized,
    self-loop-augmented operator D^{-1/2} (A + I) D^{-1/2}. Station rows follow
    ``station_order``, which must match the station order of the data arrays.

    Returns a dict whose ``adjacency`` value is the ``[N, N]`` float32
    renormalized adjacency.
    """
    table = pd.read_csv(stations_csv).set_index("station_id")
    missing = [s for s in station_order if s not in table.index]
    if missing:
        raise ValueError(f"stations missing from {stations_csv}: {missing[:5]}")
    table = table.loc[station_order]

    distances = haversine_matrix(table["lon"].values, table["lat"].values)
    adjacency = (distances <= threshold_km).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)

    augmented = adjacency + np.eye(len(station_order), dtype=np.float32)
    degree = np.clip(augmented.sum(axis=1), 1e-8, None)
    scale = 1.0 / np.sqrt(degree)
    renormalized = augmented * scale[:, None] * scale[None, :]

    return {"adjacency": torch.from_numpy(renormalized)}
