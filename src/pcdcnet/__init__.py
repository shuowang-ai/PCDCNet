"""PCDCNet: physics-informed graph-recurrent station-level air-quality forecasting."""

__version__ = "1.0.0"

from .data import DEFAULT_SPLITS, METEOROLOGY, POLLUTANTS, KnowAirV2, Normalization
from .graph import build_station_graph, haversine_matrix
from .loss import prediction_loss
from .metrics import evaluate
from .model import PCDCNet

__all__ = [
    "DEFAULT_SPLITS",
    "METEOROLOGY",
    "POLLUTANTS",
    "KnowAirV2",
    "Normalization",
    "PCDCNet",
    "build_station_graph",
    "evaluate",
    "haversine_matrix",
    "prediction_loss",
]
