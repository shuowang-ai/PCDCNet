#!/usr/bin/env python
"""Download the KnowAir-V2 dataset from Zenodo and verify its checksums.

KnowAir-V2: https://doi.org/10.5281/zenodo.15614907 (CC BY 4.0).

    python scripts/download_data.py --output data/knowair_v2 [--region bthsa yrd]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests

RECORD_URL = "https://zenodo.org/api/records/15614907"

EXPECTED_MD5 = {
    "dataset_bthsa.nc": "7c738d99246683c2711ac48f3a737aa9",
    "stations_bthsa.csv": "a7fd2302a5b21276276e78c9355af764",
    "dataset_yrd.nc": "7c7dcef8b5070cbff3ec9abc198a0fb9",
    "stations_yrd.csv": "1a23d70eb49fd0a350d475518089d45e",
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/knowair_v2")
    parser.add_argument("--region", nargs="+", default=["bthsa", "yrd"], choices=["bthsa", "yrd"])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    wanted = {name for name in EXPECTED_MD5 if any(r in name for r in args.region)}
    record = requests.get(RECORD_URL, timeout=60)
    record.raise_for_status()
    try:
        files = {entry["key"]: entry for entry in record.json()["files"]}
        missing = wanted - files.keys()
        if missing:
            raise KeyError(sorted(missing))
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"unexpected Zenodo record layout at {RECORD_URL}: {error}; "
            "download the files manually from https://doi.org/10.5281/zenodo.15614907"
        ) from error

    for name in sorted(wanted):
        destination = output / name
        if destination.exists() and md5sum(destination) == EXPECTED_MD5[name]:
            print(f"{name}: already present, checksum OK")
            continue
        url = files[name]["links"]["self"]
        print(f"{name}: downloading {files[name]['size']} bytes ...")
        partial = destination.with_suffix(destination.suffix + ".part")
        with requests.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(1 << 20):
                    handle.write(chunk)
        partial.replace(destination)
        digest = md5sum(destination)
        if digest != EXPECTED_MD5[name]:
            raise RuntimeError(f"{name}: checksum mismatch ({digest})")
        print(f"{name}: checksum OK")


if __name__ == "__main__":
    main()
