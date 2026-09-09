#!/usr/bin/env python3
"""Within-subject experiment entry point for optional MOABB datasets."""

from __future__ import annotations

import argparse

import yaml

from mi_eeg_qa.data.moabb_loaders import MOABBRequest, load_moabb_epochs


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a within-subject MOABB dataset (optional dependency path).")
    parser.add_argument("--config", default="configs/bcic2a.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    request = MOABBRequest(dataset=ds["name"], subjects=tuple(ds.get("subjects", [1])), paradigm=ds.get("paradigm", "left_right_hand"))
    X, y, metadata = load_moabb_epochs(request)
    print(f"Loaded {request.dataset}: X={X.shape}, y={len(y)}, metadata_rows={len(metadata)}")
    print("Training/evaluation wiring is intentionally left to protocol-specific extensions.")


if __name__ == "__main__":
    main()
