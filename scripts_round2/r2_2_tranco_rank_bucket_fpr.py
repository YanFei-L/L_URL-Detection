import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


def _default_paths(base_dir: str) -> dict:
    return {
        "top1m": os.path.join(base_dir, "data", "top-1m.csv"),
        "model": os.path.join(base_dir, "models", "XGBoost.joblib"),
        "output": os.path.join(base_dir, "round2_tranco_rank_bucket_fpr.json"),
        "feature_extraction_dir": os.path.join(base_dir, "feature_extraction"),
    }


def _parse_bucket(s: str) -> Tuple[str, int, int]:
    parts = s.replace(" ", "").split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid bucket spec: {s}. Expected NAME:START-END")
    name, rng = parts[0], parts[1]
    if "-" not in rng:
        raise ValueError(f"Invalid bucket range: {s}. Expected START-END")
    start_s, end_s = rng.split("-", 1)
    start, end = int(start_s), int(end_s)
    if start <= 0 or end <= 0 or end < start:
        raise ValueError(f"Invalid bucket range values: {s}")
    return name, start, end


def _read_domains_by_rank_ranges(top1m_path: str, buckets: Dict[str, Tuple[int, int]]) -> Dict[str, List[str]]:
    max_end = max(end for _, end in buckets.values())
    out: Dict[str, List[str]] = {k: [] for k in buckets.keys()}

    with open(top1m_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                rank = int(row[0])
            except Exception:
                continue
            if rank > max_end:
                break
            domain = str(row[1]).strip()
            for name, (start, end) in buckets.items():
                if start <= rank <= end:
                    out[name].append(domain)

    return out


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> int:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    defaults = _default_paths(base_dir)

    parser = argparse.ArgumentParser(description="R2-2: Tranco rank-bucket false-positive analysis (no retraining).")
    parser.add_argument("--top1m", default=defaults["top1m"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--output", default=defaults["output"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["bucket1:1-40000", "bucket2:200000-240000", "bucket3:600000-640000"],
        help="Bucket specs, e.g. bucket1:1-40000 bucket2:200000-240000",
    )

    args = parser.parse_args()

    sys.path.insert(0, defaults["feature_extraction_dir"])
    try:
        from feature_extractor import FeatureExtractor  # type: ignore
    except Exception as e:
        raise ImportError(f"Failed to import FeatureExtractor from {defaults['feature_extraction_dir']}: {e}")

    buckets: Dict[str, Tuple[int, int]] = {}
    for spec in args.buckets:
        name, start, end = _parse_bucket(spec)
        buckets[name] = (start, end)

    domains_by_bucket = _read_domains_by_rank_ranges(args.top1m, buckets)

    model = joblib.load(args.model)
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    extractor = FeatureExtractor()

    selected_features = ["url_length", "path_length", "hostname_entropy"]

    out: Dict = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "threshold": float(args.threshold),
            "paths": {
                "top1m": os.path.relpath(args.top1m, base_dir),
                "model": os.path.relpath(args.model, base_dir),
            },
            "buckets": {k: {"start": v[0], "end": v[1]} for k, v in buckets.items()},
            "label_convention": {"predicted_label": {"0": "Malicious", "1": "Benign"}},
        },
        "results": {},
    }

    for bucket_name, domains in domains_by_bucket.items():
        urls = [f"https://{d}" for d in domains]
        rows: List[Dict] = []
        for u in urls:
            feats = extractor.extract_all_features(u)
            rows.append(feats)

        if not rows:
            out["results"][bucket_name] = {"samples": 0, "fpr": None}
            continue

        X = pd.DataFrame(rows)
        y_prob_pos1 = model.predict_proba(X)[:, 1]
        y_pred = (y_prob_pos1 >= float(args.threshold)).astype(int)

        fpr = float(np.mean(y_pred == 0))

        feature_stats = {}
        for col in selected_features:
            if col in X.columns:
                vals = X[col].to_numpy(dtype=float)
                feature_stats[col] = {
                    "mean": _safe_float(np.mean(vals)),
                    "median": _safe_float(np.median(vals)),
                }

        out["results"][bucket_name] = {
            "samples": int(len(urls)),
            "fpr": fpr,
            "feature_stats": feature_stats,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
