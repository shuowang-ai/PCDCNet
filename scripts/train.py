#!/usr/bin/env python
"""Train the emission-free PCDCNet configuration on one KnowAir-V2 region.

Example (full reproduction, after downloading the data):

    python scripts/train.py \
        --dataset data/knowair_v2/dataset_bthsa.nc \
        --stations data/knowair_v2/stations_bthsa.csv \
        --output checkpoints/bthsa

Example (bundled one-month sample, pipeline demonstration only):

    python scripts/train.py --demo --max-epochs 3 \
        --dataset data/sample/knowair_v2_bthsa_2017-01.nc \
        --stations data/sample/stations_bthsa.csv \
        --output runs/demo
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import _path  # noqa: F401  (makes src/ importable when run from a checkout)
import numpy as np
import torch
from torch.utils.data import DataLoader

from pcdcnet import (
    KnowAirV2,
    PCDCNet,
    build_station_graph,
    evaluate,
    prediction_loss,
)

DEMO_SPLITS = {"train": (2017, 2017), "val": (2017, 2017), "test": (2017, 2017)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="KnowAir-V2 regional NetCDF file")
    parser.add_argument("--stations", required=True, help="matching station-coordinate CSV")
    parser.add_argument(
        "--emissions",
        help="optional NetCDF with user-prepared emission channels on the same "
        "[time, station] grid (MEIC must be requested from its provider; see README)",
    )
    parser.add_argument("--output", required=True, help="output directory for checkpoints")
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulate", type=int, default=4, help="gradient-accumulation steps")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10, help="early-stopping patience")
    parser.add_argument("--clip-norm", type=float, default=2.0, help="gradient-norm clip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="use the bundled sample's within-month splits (pipeline demo, not a benchmark)",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_kwargs = {}
    if args.demo:
        # The sample covers one month: slice it into contiguous thirds by hours.
        import xarray as xr

        with xr.open_dataset(args.dataset) as ds:
            year = int(ds["time"].dt.year.values[0])
        split_kwargs["splits"] = {k: (year, year) for k in ("train", "val", "test")}

    train_set = KnowAirV2(
        args.dataset,
        "train",
        args.history,
        args.horizon,
        emissions_path=args.emissions,
        **split_kwargs,
    )
    if args.demo:
        # Reuse the training-month statistics; carve demo val/test from the tail.
        total = len(train_set)
        train_windows = range(int(total * 0.7))
        val_windows = range(int(total * 0.7), int(total * 0.85))
        test_windows = range(int(total * 0.85), total)
        from torch.utils.data import Subset

        val_set = Subset(train_set, list(val_windows))
        eval_reference = train_set
        eval_indices = list(test_windows)
        train_view = Subset(train_set, list(train_windows))
    else:
        val_set = KnowAirV2(
            args.dataset,
            "val",
            args.history,
            args.horizon,
            normalization=train_set.normalization,
            emissions_path=args.emissions,
        )
        eval_reference = KnowAirV2(
            args.dataset,
            "test",
            args.history,
            args.horizon,
            normalization=train_set.normalization,
            emissions_path=args.emissions,
        )
        eval_indices = None
        train_view = train_set

    graph = build_station_graph(args.stations, train_set.stations)
    adjacency = graph["adjacency"].to(args.device)

    model = PCDCNet(
        num_pollutants=train_set.pollutants.shape[-1],
        num_covariates=train_set.covariates.shape[-1],
        hidden_size=args.hidden_size,
    ).to(args.device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())}", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=3)
    autocast = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=autocast)

    train_loader = DataLoader(
        train_view,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)
    if len(train_loader) == 0:
        raise ValueError("training split has fewer windows than --batch-size; use a smaller batch")
    if args.accumulate < 1:
        raise ValueError("--accumulate must be positive")

    def validation_mae() -> float:
        """Mean of minibatch MAEs in standardized units (model-selection criterion)."""
        model.eval()
        total, batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                output = model(
                    batch["pollutant_history"].to(args.device),
                    batch["covariate_history"].to(args.device),
                    batch["covariate_future"].to(args.device),
                    adjacency,
                )
                total += prediction_loss(
                    output["prediction"], batch["pollutant_future"].to(args.device)
                ).item()
                batches += 1
        return total / max(batches, 1)

    best_val, best_epoch = float("inf"), -1
    checkpoint_path = out_dir / "model.pt"
    history = []

    for epoch in range(args.max_epochs):
        model.train()
        started = time.time()
        running_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            with torch.autocast(args.device.split(":")[0], torch.float16, enabled=autocast):
                output = model(
                    batch["pollutant_history"].to(args.device),
                    batch["covariate_history"].to(args.device),
                    batch["covariate_future"].to(args.device),
                    adjacency,
                )
                loss = prediction_loss(
                    output["prediction"], batch["pollutant_future"].to(args.device)
                )
            # The final accumulation group can be shorter than --accumulate.
            group_start = (step // args.accumulate) * args.accumulate
            group_size = min(args.accumulate, len(train_loader) - group_start)
            scaler.scale(loss / group_size).backward()
            if (step + 1) % args.accumulate == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item()
            steps += 1

        val_mae = validation_mae()
        scheduler.step(val_mae)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / steps,
                "val_mae": val_mae,
                "seconds": round(time.time() - started, 1),
            }
        )
        print(
            f"epoch {epoch:3d}  train {running_loss / steps:.4f}  "
            f"val {val_mae:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}  "
            f"{history[-1]['seconds']}s",
            flush=True,
        )

        if math.isfinite(val_mae) and val_mae < best_val:
            best_val, best_epoch = val_mae, epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "normalization": train_set.normalization.state_dict(),
                    "stations": train_set.stations,
                    "config": {
                        "history": args.history,
                        "horizon": args.horizon,
                        "hidden_size": args.hidden_size,
                        "seed": args.seed,
                        "covariates": train_set.covariate_names,
                        "objective": "mean absolute error",
                        "validation_aggregation": "mean of minibatch standardized MAEs",
                        "normalization_fit": "float64 accumulation, stored as float32",
                    },
                },
                str(checkpoint_path) + ".tmp",
            )
            os.replace(str(checkpoint_path) + ".tmp", checkpoint_path)
        elif epoch - best_epoch >= args.patience:
            print(f"early stop at epoch {epoch} (best {best_val:.4f} @ {best_epoch})", flush=True)
            break

    state = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    model.load_state_dict(state["model"])
    metrics = evaluate(
        model,
        eval_reference,
        graph["adjacency"],
        args.batch_size,
        args.device,
        args.num_workers,
        window_indices=eval_indices,
    )
    if args.demo:
        metrics["note"] = "demo split on the bundled sample; not a benchmark result"
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_mae": best_val,
                "best_epoch": best_epoch,
                "test": metrics,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({k: metrics[k] for k in ("MAE", "RMSE")}, indent=2))


if __name__ == "__main__":
    main()
