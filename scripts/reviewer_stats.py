import argparse
import json
import os
import platform
import sys

import pandas as pd


def _safe_import(name: str):
    try:
        module = __import__(name)
        return module, None
    except Exception as e:
        return None, str(e)


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def collect_environment_info() -> dict:
    info = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }

    xgb, xgb_err = _safe_import("xgboost")
    info["xgboost_version"] = getattr(xgb, "__version__", None) if xgb else None
    info["xgboost_import_error"] = xgb_err

    shap, shap_err = _safe_import("shap")
    info["shap_version"] = getattr(shap, "__version__", None) if shap else None
    info["shap_import_error"] = shap_err

    sklearn, sklearn_err = _safe_import("sklearn")
    info["sklearn_version"] = getattr(sklearn, "__version__", None) if sklearn else None
    info["sklearn_import_error"] = sklearn_err

    pd_mod, pd_err = _safe_import("pandas")
    info["pandas_version"] = getattr(pd_mod, "__version__", None) if pd_mod else None
    info["pandas_import_error"] = pd_err

    return info


def extract_xgboost_hyperparams_from_model(model_path: str) -> dict:
    joblib, joblib_err = _safe_import("joblib")
    if not joblib:
        return {"available": False, "reason": f"joblib import failed: {joblib_err}"}

    if not os.path.exists(model_path):
        return {"available": False, "reason": f"model not found: {model_path}"}

    model = joblib.load(model_path)

    result = {
        "available": True,
        "model_path": model_path,
    }

    try:
        booster = model.get_booster()
        config = json.loads(booster.save_config())

        tree_train_param = (
            config.get("learner", {})
            .get("gradient_booster", {})
            .get("tree_train_param", {})
        )

        objective = config.get("learner", {}).get("objective", {}).get("name")
        result["objective"] = objective

        xgb_params = {
            "n_estimators": booster.num_boosted_rounds(),
            "max_depth": tree_train_param.get("max_depth"),
            "learning_rate_eta": tree_train_param.get("eta"),
            "subsample": tree_train_param.get("subsample"),
            "colsample_bytree": tree_train_param.get("colsample_bytree"),
            "min_child_weight": tree_train_param.get("min_child_weight"),
            "gamma": tree_train_param.get("gamma"),
            "reg_lambda": tree_train_param.get("lambda"),
            "reg_alpha": tree_train_param.get("alpha"),
        }

        try:
            xgb_params["eval_metric"] = model.get_params().get("eval_metric")
            xgb_params["random_state"] = model.get_params().get("random_state")
        except Exception:
            pass

        result["hyperparameters"] = xgb_params
        return result
    except Exception as e:
        return {"available": False, "reason": f"failed to parse booster config: {e}"}


def compute_url_dedup_stats(white_list_path: str, block_list_path: str) -> dict:
    white = _read_csv(white_list_path)
    block = _read_csv(block_list_path)

    if "url" not in white.columns or "url" not in block.columns:
        raise KeyError("white_list.csv and block_list.csv must contain a 'url' column")

    white_urls = white["url"].astype(str)
    block_urls = block["url"].astype(str)

    white_unique = int(white_urls.nunique(dropna=True))
    block_unique = int(block_urls.nunique(dropna=True))

    combined = pd.concat(
        [
            pd.DataFrame({"url": white_urls, "src": "white"}),
            pd.DataFrame({"url": block_urls, "src": "block"}),
        ],
        ignore_index=True,
    )

    combined_unique = int(combined["url"].nunique(dropna=True))
    cross_source_duplicates = int((combined.groupby("url")["src"].nunique() > 1).sum())

    return {
        "white_rows": int(len(white)),
        "white_unique": white_unique,
        "block_rows": int(len(block)),
        "block_unique": block_unique,
        "combined_rows": int(len(combined)),
        "combined_unique": combined_unique,
        "duplicates_total": int(len(combined) - combined_unique),
        "duplicates_within_block": int(len(block) - block_unique),
        "duplicates_within_white": int(len(white) - white_unique),
        "cross_source_duplicates": cross_source_duplicates,
    }


def compute_phishtank_target_distribution(verified_online_path: str, n: int = 40000, top_k: int = 10) -> dict:
    df = _read_csv(verified_online_path)

    if "target" not in df.columns:
        raise KeyError("verified_online.csv must contain a 'target' column")

    df = df.head(n)
    vc = df["target"].fillna("NA").astype(str).value_counts()

    total = int(vc.sum())
    top = vc.head(top_k)
    remaining_count = int(total - int(top.sum()))
    distinct = int(vc.size)

    top_rows = []
    for name, count in top.items():
        top_rows.append(
            {
                "target": name,
                "count": int(count),
                "share_percent": float(count) / float(total) * 100.0 if total else 0.0,
            }
        )

    return {
        "total_rows": total,
        "distinct_targets": distinct,
        "top_k": top_k,
        "top_targets": top_rows,
        "remaining_targets_count": max(distinct - len(top_rows), 0),
        "remaining_rows": remaining_count,
        "remaining_share_percent": float(remaining_count) / float(total) * 100.0 if total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        help="Project root directory (default: repo root inferred from this script)",
    )
    parser.add_argument(
        "--output-json",
        default="reviewer_stats.json",
        help="Output JSON path (relative to project root unless absolute)",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)

    paths = {
        "white_list": os.path.join(project_root, "data", "white_list.csv"),
        "block_list": os.path.join(project_root, "data", "block_list.csv"),
        "verified_online": os.path.join(project_root, "data", "verified_online.csv"),
        "xgb_model": os.path.join(project_root, "models", "XGBoost.joblib"),
    }

    payload = {
        "project_root": project_root,
        "paths": paths,
        "environment": collect_environment_info(),
        "xgboost_model": extract_xgboost_hyperparams_from_model(paths["xgb_model"]),
        "dataset_dedup": compute_url_dedup_stats(paths["white_list"], paths["block_list"]),
        "phishtank_target_distribution": compute_phishtank_target_distribution(paths["verified_online"]),
    }

    out_path = args.output_json
    if not os.path.isabs(out_path):
        out_path = os.path.join(project_root, out_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
