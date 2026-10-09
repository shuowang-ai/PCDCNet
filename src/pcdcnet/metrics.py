"""Evaluation metrics in physical units.

Errors are computed per forecast lead after inverting the channel-wise
standardization, then summarized per pollutant by averaging over the leads:

* ``MAE``  = mean over leads of the per-lead mean absolute error,
* ``RMSE`` = mean over leads of the per-lead root-mean-square error.

This lead-averaged convention matches the regional summaries reported for the
neural models in the PCDCNet manuscript.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import POLLUTANTS, KnowAirV2
from .model import PCDCNet


@torch.no_grad()
def evaluate(
    model: PCDCNet,
    dataset: KnowAirV2,
    adjacency: torch.Tensor,
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 2,
    window_indices: list[int] | None = None,
) -> dict:
    """Return per-pollutant lead-averaged MAE and RMSE plus per-lead curves.

    ``window_indices`` restricts scoring to a subset of the dataset's windows
    (used by the sample demo); by default every window is scored.
    """
    model = model.to(device).eval()
    adjacency = adjacency.to(device)
    windows: torch.utils.data.Dataset = dataset
    if window_indices is not None:
        windows = torch.utils.data.Subset(dataset, window_indices)
    loader = DataLoader(windows, batch_size=batch_size, num_workers=num_workers)

    horizon = dataset.horizon
    channels = len(POLLUTANTS)
    abs_sum = torch.zeros(horizon, channels, dtype=torch.float64)
    sq_sum = torch.zeros(horizon, channels, dtype=torch.float64)
    count = torch.zeros(horizon, channels, dtype=torch.float64)

    for batch in loader:
        output = model(
            batch["pollutant_history"].to(device),
            batch["covariate_history"].to(device),
            batch["covariate_future"].to(device),
            adjacency,
        )
        prediction = dataset.denormalize_pollutants(output["prediction"])
        target = dataset.denormalize_pollutants(batch["pollutant_future"].to(device))
        error = (prediction - target).double()
        abs_sum += error.abs().sum(dim=(0, 2)).cpu()
        sq_sum += (error**2).sum(dim=(0, 2)).cpu()
        count += error.shape[0] * error.shape[2]

    mae_per_lead = (abs_sum / count).numpy()
    rmse_per_lead = np.sqrt((sq_sum / count).numpy())
    return {
        "pollutants": list(POLLUTANTS),
        "MAE": {p: float(mae_per_lead[:, c].mean()) for c, p in enumerate(POLLUTANTS)},
        "RMSE": {p: float(rmse_per_lead[:, c].mean()) for c, p in enumerate(POLLUTANTS)},
        "MAE_per_lead": mae_per_lead.tolist(),
        "RMSE_per_lead": rmse_per_lead.tolist(),
        "num_windows": len(windows),
    }
