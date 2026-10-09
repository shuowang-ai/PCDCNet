#!/usr/bin/env python
"""Evaluate a trained PCDCNet checkpoint on a KnowAir-V2 test split.

    python scripts/evaluate.py \
        --checkpoint checkpoints/bthsa/model.pt \
        --dataset data/knowair_v2/dataset_bthsa.nc \
        --stations data/knowair_v2/stations_bthsa.csv
"""

from __future__ import annotations

import argparse
import json

import _path  # noqa: F401
import torch

from pcdcnet import KnowAirV2, Normalization, PCDCNet, build_station_graph, evaluate
from pcdcnet.data import POLLUTANTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stations", required=True)
    parser.add_argument("--emissions", help="optional emission NetCDF (see README)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="optional JSON file for the metrics")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    normalization = Normalization.from_state_dict(state["normalization"])

    dataset = KnowAirV2(
        args.dataset,
        args.split,
        history=config["history"],
        horizon=config["horizon"],
        normalization=normalization,
        emissions_path=args.emissions,
        covariate_names=config.get("covariates"),
    )
    if dataset.stations != state["stations"]:
        raise RuntimeError("station order of the dataset does not match the checkpoint")

    graph = build_station_graph(args.stations, dataset.stations)
    model = PCDCNet(
        num_pollutants=len(POLLUTANTS),
        num_covariates=len(normalization.covariate_mean),
        hidden_size=config["hidden_size"],
    )
    model.load_state_dict(state["model"])

    metrics = evaluate(model, dataset, graph["adjacency"], args.batch_size, args.device)
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_windows": metrics["num_windows"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(
                {
                    **summary,
                    "MAE_per_lead": metrics["MAE_per_lead"],
                    "RMSE_per_lead": metrics["RMSE_per_lead"],
                },
                handle,
                indent=2,
            )
            handle.write("\n")


if __name__ == "__main__":
    main()
