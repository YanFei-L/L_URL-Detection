import argparse
import json
import os
from datetime import datetime
from typing import Dict, Optional

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix


def _default_paths(base_dir: str) -> dict:
    return {
        "test_data": os.path.join(base_dir, "data", "test_data.csv"),
        "verified_online": os.path.join(base_dir, "data", "verified_online.csv"),
        "model": os.path.join(base_dir, "models", "XGBoost.joblib"),
        "output": os.path.join(base_dir, "round2_target_subgroup_eval.json"),
    }


def _load_url_to_target(verified_online_path: str, limit_rows: Optional[int]) -> Dict[str, str]:
    df = pd.read_csv(verified_online_path)
    if limit_rows is not None:
        df = df.head(int(limit_rows))

    if "url" not in df.columns or "target" not in df.columns:
        raise KeyError("verified_online.csv must contain 'url' and 'target' columns")

    url_to_target: Dict[str, str] = {}
    for url, target in zip(df["url"].astype(str), df["target"].astype(str)):
        url_to_target[url.strip()] = target.strip()

    return url_to_target


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d != 0 else 0.0


def main() -> int:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    defaults = _default_paths(base_dir)

    parser = argparse.ArgumentParser(description="R2-1: subgroup evaluation by PhishTank target (Other vs Non-Other) on malicious samples.")
    parser.add_argument("--test-data", default=defaults["test_data"])
    parser.add_argument("--verified-online", default=defaults["verified_online"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--output", default=defaults["output"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--verified-limit-rows", type=int, default=40000)
    parser.add_argument("--other-label", default="Other")

    args = parser.parse_args()

    test_df = pd.read_csv(args.test_data)
    if "url" not in test_df.columns or "label" not in test_df.columns:
        raise KeyError("test_data.csv must contain 'url' and 'label' columns")

    X = test_df.drop(columns=["url", "label"])
    y_true = test_df["label"].astype(int).to_numpy()
    urls = test_df["url"].astype(str).map(lambda s: s.strip()).to_numpy()

    model = joblib.load(args.model)
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    y_prob_pos1 = model.predict_proba(X)[:, 1]
    y_pred = (y_prob_pos1 >= float(args.threshold)).astype(int)

    url_to_target = _load_url_to_target(args.verified_online, args.verified_limit_rows)
    targets = [url_to_target.get(u, "") for u in urls]

    is_mal = (y_true == 0)
    y_true_mal = y_true[is_mal]
    y_pred_mal = y_pred[is_mal]
    targets_mal = [targets[i] for i in range(len(targets)) if bool(is_mal[i])]

    def _group_of_target(t: str) -> str:
        if (t or "").strip() == args.other_label:
            return "Other"
        return "Non-Other"

    groups = [_group_of_target(t) for t in targets_mal]

    out: dict = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "threshold": float(args.threshold),
            "other_label": args.other_label,
            "paths": {
                "test_data": os.path.relpath(args.test_data, base_dir),
                "verified_online": os.path.relpath(args.verified_online, base_dir),
                "model": os.path.relpath(args.model, base_dir),
            },
            "verified_limit_rows": int(args.verified_limit_rows) if args.verified_limit_rows is not None else None,
        },
        "overall": {},
        "by_group": {},
        "diagnostics": {},
    }

    cm_all = confusion_matrix(y_true_mal, y_pred_mal, labels=[0, 1])
    tp = int(cm_all[0, 0])
    fn = int(cm_all[0, 1])
    out["overall"] = {
        "malicious_samples": int(len(y_true_mal)),
        "tp_malicious_pred0": tp,
        "fn_malicious_pred1": fn,
        "tpr_malicious": _safe_div(tp, tp + fn),
        "missed_threat_rate": _safe_div(fn, tp + fn),
    }

    missing_target_count = sum(1 for t in targets_mal if (t or "").strip() == "")
    out["diagnostics"]["missing_target_count_in_malicious_test"] = int(missing_target_count)

    for group_name in sorted(set(groups)):
        idx = [i for i, g in enumerate(groups) if g == group_name]
        yt = y_true_mal[idx]
        yp = y_pred_mal[idx]

        cm = confusion_matrix(yt, yp, labels=[0, 1])
        g_tp = int(cm[0, 0])
        g_fn = int(cm[0, 1])

        out["by_group"][group_name] = {
            "samples": int(len(idx)),
            "tp_malicious_pred0": g_tp,
            "fn_malicious_pred1": g_fn,
            "tpr_malicious": _safe_div(g_tp, g_tp + g_fn),
            "missed_threat_rate": _safe_div(g_fn, g_tp + g_fn),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
