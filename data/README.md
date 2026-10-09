# Data

## `sample/`

`knowair_v2_bthsa_2017-01.nc` is a one-month subset (January 2017, 228
stations) of the KnowAir-V2 BTHSA regional file, bundled so the pipeline runs
out of the box. `stations_bthsa.csv` is the unmodified station table.

Source: KnowAir-V2, https://doi.org/10.5281/zenodo.15614907, CC BY 4.0.
The subset is for pipeline demonstration only; download the full record for
any meaningful training or evaluation:

    uv run python scripts/download_data.py --output data/knowair_v2

## `knowair_v2/` (not committed)

Target directory for the full download (about 1 GB). The download script
verifies the Zenodo MD5 checksums.

MEIC emission inventories are not redistributable here; see the README
section "Optional: MEIC emission features".
