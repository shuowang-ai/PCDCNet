# Reproduction

## Environment

```bash
uv sync --extra cu128   # or --extra cpu
uv run pytest -q
```

The tests cover forward/backward computation, optional emissions, named checkpoint channels, evaluation and prediction CLIs, and training with an incomplete gradient-accumulation group.

## 1. Pipeline demo (no download)

```bash
uv run python scripts/train.py --demo --max-epochs 8 \
    --dataset data/sample/knowair_v2_bthsa_2017-01.nc \
    --stations data/sample/stations_bthsa.csv \
    --output runs/demo
```

The demo divides the month's overlapping windows into 70% training, 15% validation and 15% test, and uses normalization from the entire sample month. Windows at the boundaries share hours. This demonstrates the pipeline; its scores are not a leakage-free benchmark or comparable to the full-year experiment.

## 2. Re-scoring released checkpoints

A checkpoint package is complete when it contains both `model.pt` and `metrics.json`.

```bash
uv run python scripts/download_data.py --output data/knowair_v2
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/bthsa/model.pt \
    --dataset data/knowair_v2/dataset_bthsa.nc \
    --stations data/knowair_v2/stations_bthsa.csv --device cpu
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/yrd/model.pt \
    --dataset data/knowair_v2/dataset_yrd.nc \
    --stations data/knowair_v2/stations_yrd.csv --device cpu
```

The checkpoint controls history length, channel names/order, model dimensions and normalization. `torch.load(..., weights_only=True)` loads all released checkpoints.

All results use the full 2022–2023 test split and every valid hourly forecast window, with no concentration clipping or random graph dropout. MAE and RMSE are calculated per lead in µg/m³, aggregating all windows and stations before averaging the 72 per-lead values. Thus lead-averaged RMSE differs from pooled RMSE. `--output metrics.json` additionally saves all per-lead metrics. The `num_windows` value also verifies complete evaluation coverage: 17,425 windows per region.

Compare the printed scores with `checkpoints/<name>/metrics.json` (`test.MAE` and `test.RMSE`). CPU and GPU floating-point calculations can differ slightly; exact equality is only expected with the same checkpoint, inputs, device, batch size and numerical environment. Changing hardware or the software stack is not an exact-reproduction guarantee.

## 3. Retraining from scratch

```bash
uv run python scripts/train.py \
    --dataset data/knowair_v2/dataset_bthsa.nc \
    --stations data/knowair_v2/stations_bthsa.csv \
    --output runs/bthsa
```

Use the YRD dataset and station table with `--output runs/yrd` for the other region. Full-data training splits by calendar years: 2016–2019 for training, 2020–2021 for validation, and 2022–2023 for test. Normalization is fitted only on the training years. Defaults are 24 h history, 72 h horizon, hidden width 32, mean absolute prediction loss, Adam at 0.001 with weight decay 0.0001, batch size 32, four-step accumulation, gradient-norm clipping at 2.0, seed 42 and mixed precision on CUDA.

The scheduler and early stopping use the unweighted mean of validation minibatch MAEs in standardized coordinates, including the final short batch. This is a model-selection criterion, distinct from the physical-unit test scores. Early stopping has patience 10; the learning-rate scheduler multiplies the rate by 0.3 after its patience of 3. `metrics.json` records the best validation value, selected epoch, epoch history and full test metrics. These are single-seed runs; retraining can differ due to hardware, numerical nondeterminism and software versions.

Training uses float64 accumulation for channel means and standard deviations, stored as float32. Evaluation and prediction use each checkpoint's saved statistics rather than refitting.

## 4. Prediction

```bash
uv run python scripts/predict.py \
    --checkpoint checkpoints/bthsa/model.pt \
    --dataset data/knowair_v2/dataset_bthsa.nc \
    --stations data/knowair_v2/stations_bthsa.csv \
    --init-time 2023-06-01T00:00 --output forecast.csv
```

The timestamp is the first forecast valid time. Observed pollutants are required only for the preceding history; meteorological inputs cover history and all future leads. ERA5 in the public dataset supports retrospective evaluation, not a simulation of meteorological information available at issuance. For a checkpoint trained with optional emissions, also pass the matching `--emissions` NetCDF to evaluation and prediction, as for training.

## Relation to the manuscript

The released checkpoints train this implementation from scratch with eight meteorological covariates and 24 h history. The public dataset and training pipeline reproduce the released emission-free experiment; licensed emission inputs, original experiment archives, baselines and production-serving code are outside this repository. The manuscript documents the upstream missing-mask and preprocessing provenance limits; fitting normalization on training years does not establish that the upstream data processing was split-independent.
