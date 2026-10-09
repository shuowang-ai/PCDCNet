"""KnowAir-V2 data access.

The KnowAir-V2 release (https://doi.org/10.5281/zenodo.15614907, CC BY 4.0)
contains one NetCDF file per region with hourly station-level variables on a
``[time, station]`` grid for 2016--2023:

* pollutants: ``PM2.5``, ``O3`` (ug/m3)
* ERA5 meteorology: ``t2m``, ``d2m``, ``sp``, ``tp``, ``blh``, ``msdwswrf``,
  ``u100``, ``v100``

plus one station-coordinate CSV per region. This module slices the arrays into
(history, horizon) windows, standardizes every channel with statistics
computed on the training years only, and exposes PyTorch datasets whose
windows lie entirely inside their split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

POLLUTANTS = ["PM2.5", "O3"]
METEOROLOGY = ["t2m", "d2m", "sp", "tp", "blh", "msdwswrf", "u100", "v100"]

#: Emission species of the original experiments (MEIC inventory channels).
#: MEIC is not redistributable with this repository; see the README for how
#: to request it and prepare these channels yourself.
EMISSION_SPECIES = ["PM2.5", "PM10", "NOx", "VOC", "NH3", "SO2"]

DEFAULT_SPLITS = {
    "train": (2016, 2019),
    "val": (2020, 2021),
    "test": (2022, 2023),
}


@dataclass
class Normalization:
    """Per-channel mean and standard deviation from the training split."""

    pollutant_mean: np.ndarray
    pollutant_std: np.ndarray
    covariate_mean: np.ndarray
    covariate_std: np.ndarray

    def state_dict(self) -> dict:
        return {k: torch.from_numpy(v.copy()) for k, v in vars(self).items()}

    @classmethod
    def from_state_dict(cls, state: dict) -> Normalization:
        return cls(**{k: np.asarray(v) for k, v in state.items()})


class KnowAirV2(Dataset):
    """Windows of one KnowAir-V2 region for one split.

    Each item provides standardized float32 tensors:
        ``pollutant_history``: [history, stations, 2]
        ``covariate_history``: [history, stations, 8]
        ``covariate_future``:  [horizon, stations, 8]
        ``pollutant_future``:  [horizon, stations, 2] (training target)
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        history: int = 24,
        horizon: int = 72,
        splits: dict[str, tuple[int, int]] | None = None,
        normalization: Normalization | None = None,
        emissions_path: str | None = None,
        covariate_names: list[str] | None = None,
    ):
        self.history = history
        self.horizon = horizon
        window = history + horizon
        splits = splits or DEFAULT_SPLITS
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; available: {sorted(splits)}")

        with xr.open_dataset(dataset_path) as ds:
            years = ds["time"].dt.year.values
            times = ds["time"].values
            self.stations = [str(s) for s in ds["station"].values]
            pollutants = np.stack([ds[v].values for v in POLLUTANTS], axis=-1)
            covariates = np.stack([ds[v].values for v in METEOROLOGY], axis=-1)
        self.covariate_names = list(METEOROLOGY)

        if emissions_path is not None:
            with xr.open_dataset(emissions_path) as emis:
                if [str(s) for s in emis["station"].values] != self.stations:
                    raise ValueError("emission file station order differs from the dataset")
                if not np.array_equal(emis["time"].values, times):
                    raise ValueError("emission file time axis differs from the dataset")
                names = [v for v in EMISSION_SPECIES if v in emis.data_vars] or list(emis.data_vars)
                extra = np.stack([emis[v].values for v in names], axis=-1)
            covariates = np.concatenate([covariates, extra], axis=-1)
            self.covariate_names += [f"emission_{v}" for v in names]

        if covariate_names is not None:
            if not covariate_names or len(set(covariate_names)) != len(covariate_names):
                raise ValueError("covariate names must be nonempty and unique")
            missing = sorted(set(covariate_names) - set(self.covariate_names))
            if missing:
                raise ValueError(
                    f"dataset lacks checkpoint covariates {missing}; supply the matching "
                    "--emissions file for emission-trained checkpoints"
                )
            indices = [self.covariate_names.index(name) for name in covariate_names]
            covariates = covariates[..., indices]
            self.covariate_names = list(covariate_names)

        if not np.isfinite(pollutants).all() or not np.isfinite(covariates).all():
            raise ValueError(
                "dataset contains non-finite values; check the input NetCDF "
                "(user-prepared emission files must be gap-free and finite)"
            )
        if normalization is not None and len(normalization.covariate_mean) != covariates.shape[-1]:
            raise ValueError(
                f"normalization has {len(normalization.covariate_mean)} covariate channels "
                f"but this dataset provides {covariates.shape[-1]}; pass the same "
                "--emissions file that was used at training time"
            )

        if normalization is None:
            lo, hi = splits["train"]
            fit = (years >= lo) & (years <= hi)
            if not fit.any():
                raise ValueError("training split is empty; cannot fit normalization")
            # Millions of float32 values (especially pressure/temperature) can
            # accumulate substantial rounding error. Fit in float64, retaining
            # float32 statistics and inputs for model training and inference.
            normalization = Normalization(
                pollutant_mean=pollutants[fit]
                .mean(axis=(0, 1), dtype=np.float64)
                .astype(np.float32),
                pollutant_std=pollutants[fit].std(axis=(0, 1), dtype=np.float64).astype(np.float32),
                covariate_mean=covariates[fit]
                .mean(axis=(0, 1), dtype=np.float64)
                .astype(np.float32),
                covariate_std=covariates[fit].std(axis=(0, 1), dtype=np.float64).astype(np.float32),
            )
        self.normalization = normalization

        lo, hi = splits[split]
        member = (years >= lo) & (years <= hi)
        if not member.any():
            raise ValueError(f"split {split!r} selects no hours")
        indices = np.nonzero(member)[0]
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"split {split!r} must select a contiguous time range")
        start, stop = int(indices[0]), int(indices[-1]) + 1

        std = np.where(normalization.pollutant_std < 1e-6, 1.0, normalization.pollutant_std)
        self.pollutants = ((pollutants[start:stop] - normalization.pollutant_mean) / std).astype(
            np.float32
        )
        cov_std = np.where(normalization.covariate_std < 1e-6, 1.0, normalization.covariate_std)
        self.covariates = (
            (covariates[start:stop] - normalization.covariate_mean) / cov_std
        ).astype(np.float32)

        self.num_windows = max(0, self.pollutants.shape[0] - window + 1)
        if self.num_windows == 0:
            raise ValueError(f"split {split!r} is shorter than one {window}-hour window")

    def denormalize_pollutants(self, standardized: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.normalization.pollutant_mean, dtype=standardized.dtype).to(
            standardized.device
        )
        std = torch.as_tensor(self.normalization.pollutant_std, dtype=standardized.dtype).to(
            standardized.device
        )
        return standardized * std + mean

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        h, f = self.history, self.horizon
        return {
            "pollutant_history": torch.from_numpy(self.pollutants[index : index + h]),
            "covariate_history": torch.from_numpy(self.covariates[index : index + h]),
            "covariate_future": torch.from_numpy(self.covariates[index + h : index + h + f]),
            "pollutant_future": torch.from_numpy(self.pollutants[index + h : index + h + f]),
        }
