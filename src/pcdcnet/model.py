"""PCDCNet: a graph-recurrent network for station-level air-quality forecasting.

The network decomposes each hourly concentration update into three residual
modules, applied in order at every step:

* **LID** (Local Interaction Dynamics) — a two-layer SiLU MLP that mixes the
  concentration state with the covariates at one station.
* **STD** (Spatial Transport Dynamics) — residual graph-convolution hops over
  the station graph.
* **TAD** (Temporal Accumulation Dynamics) — a gated recurrent cell that
  carries memory across steps.

A shared linear readout predicts the concentration *increment*, so each step
advances the previous concentration state. During an initial warm-up pass over
the observed history the same modules run with observed concentrations as
input; forecasting then proceeds autoregressively with forecast-valid
covariates supplied at each step.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the channel dimension."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RecurrentCell(nn.Module):
    """Gated recurrent cell used by the TAD module.

    This is a GRU-style cell with one deliberate difference from the standard
    GRU: the reset gate multiplies the raw hidden state added to the candidate
    preactivation, rather than the recurrent affine term:

        r, z, n = chunks(W_ih x + b_ih + W_hh h + b_hh)
        candidate = tanh(n + sigmoid(r) * h)
        h' = (1 - sigmoid(z)) * candidate + sigmoid(z) * h

    This follows the retained reference implementation described in the manuscript.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(
            hidden, self.weight_hh, self.bias_hh
        )
        reset, update, candidate = gates.chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate = torch.tanh(candidate + reset * hidden)
        return (1.0 - update) * candidate + update * hidden


class GraphConvBlock(nn.Module):
    """Residual graph-convolution hops on a renormalized adjacency (STD).

    Each hop propagates features over the self-loop-augmented, symmetrically
    normalized station graph, applies a linear map, GELU and dropout, and adds
    the result back. During training a fresh DropEdge mask is sampled at each
    time step (shared by the hops within that step); evaluation always uses
    the full graph operator.
    """

    def __init__(
        self,
        hidden_size: int,
        num_hops: int = 2,
        dropout: float = 0.1,
        drop_edge: float = 0.1,
    ):
        super().__init__()
        self.hops = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(num_hops))
        self.dropout = nn.Dropout(dropout)
        self.drop_edge = drop_edge

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [batch, stations, hidden]; adjacency: [stations, stations] (renormalized).
        if self.training and self.drop_edge > 0:
            mask = (torch.rand_like(adjacency) > self.drop_edge).to(adjacency.dtype)
            adjacency = adjacency * mask
        h = x
        for linear in self.hops:
            messages = torch.einsum("mn,bnf->bmf", adjacency, h)
            h = h + self.dropout(F.gelu(linear(messages)))
        return h


class PCDCNet(nn.Module):
    """Graph-recurrent forecaster producing residual concentration updates.

    Args:
        num_pollutants: number of predicted pollutant channels (e.g. 2).
        num_covariates: number of exogenous input channels per step
            (meteorology in this release; additional channels such as
            emission features can be concatenated by the caller).
        hidden_size: width of the shared hidden representation.
        num_hops: residual graph-convolution hops in the STD module.
        dropout: dropout rate inside LID and STD.
        drop_edge: edge-dropout probability for STD during training.

    All inputs and outputs are channel-wise standardized values; callers
    invert the standardization before computing physical-unit metrics.
    """

    def __init__(
        self,
        num_pollutants: int,
        num_covariates: int,
        hidden_size: int = 32,
        num_hops: int = 2,
        dropout: float = 0.1,
        drop_edge: float = 0.1,
    ):
        super().__init__()
        self.num_pollutants = num_pollutants
        self.hidden_size = hidden_size
        input_size = num_pollutants + num_covariates

        self.embed = nn.Linear(input_size, hidden_size)

        # LID: feature mixing at each station.
        self.local_norm = RMSNorm(hidden_size)
        self.local_mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

        # STD: spatial propagation over the station graph.
        self.spatial_norm = RMSNorm(hidden_size)
        self.spatial = GraphConvBlock(hidden_size, num_hops, dropout, drop_edge)

        # TAD: temporal accumulation across steps.
        self.temporal_input_norm = RMSNorm(hidden_size)
        self.temporal_state_norm = RMSNorm(hidden_size)
        self.temporal = RecurrentCell(hidden_size, hidden_size)

        # Shared residual readout.
        self.readout_norm = RMSNorm(hidden_size)
        self.readout = nn.Linear(hidden_size, num_pollutants)

    def forward(
        self,
        pollutant_history: torch.Tensor,
        covariate_history: torch.Tensor,
        covariate_future: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run warm-up over the history, then autoregressive forecasting.

        Args:
            pollutant_history: [batch, history, stations, pollutants].
            covariate_history: [batch, history, stations, covariates].
            covariate_future: [batch, horizon, stations, covariates].
            adjacency: [stations, stations] renormalized adjacency.

        Returns:
            dict with "prediction": [batch, horizon, stations, pollutants].
        """
        history_len = pollutant_history.shape[1]
        covariates = torch.cat([covariate_history, covariate_future], dim=1)
        batch, total_len, stations, _ = covariates.shape

        hidden_state = covariates.new_zeros(batch * stations, self.hidden_size)
        concentration = pollutant_history[:, 0]
        predictions = []

        for t in range(total_len):
            if t < history_len:
                concentration = pollutant_history[:, t]

            step_input = torch.cat([covariates[:, t], concentration], dim=-1)
            features = self.embed(step_input)

            features = features + self.local_mlp(self.local_norm(features))

            spatial_branch = self.spatial(self.spatial_norm(features), adjacency)
            features = features + spatial_branch

            hidden_state = self.temporal(
                self.temporal_input_norm(features.reshape(batch * stations, -1)),
                self.temporal_state_norm(hidden_state),
            )
            features = features + hidden_state.view(batch, stations, -1)

            concentration = concentration + self.readout(self.readout_norm(features))

            if t >= history_len:
                predictions.append(concentration)

        return {"prediction": torch.stack(predictions, dim=1)}
