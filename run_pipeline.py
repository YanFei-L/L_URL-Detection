import argparse
import os
import subprocess
import sys
from datetime import datetime


def _run_step(project_dir: str, script_name: str) -> None:
    script_path = os.path.abspath(os.path.join(project_dir, script_name))
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== Running: {script_name}  (start: {started}) ===")
    subprocess.run([sys.executable, script_path], cwd=project_dir, check=True)
    finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== Finished: {script_name} (end: {finished}) ===\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Project root directory (default: this file's directory)",
    )
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-feature", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-charts", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")

    args = parser.parse_args()
    project_dir = os.path.abspath(args.project_dir)

    steps = []
    if not args.skip_preprocess:
        steps.append(os.path.join("data", "data_preprocess.py"))
    if not args.skip_feature:
        steps.append(os.path.join("feature_extraction", "run_feature_extraction.py"))
    if not args.skip_train:
        steps.append(os.path.join("model_training", "train_models.py"))
    if not args.skip_charts:
        steps.append(os.path.join("model_training", "generate_charts.py"))
    if not args.skip_shap:
        steps.append(os.path.join("SHAP", "explain_models.py"))
    if args.run_benchmark:
        steps.append(os.path.join("efficiency_benchmark", "simple_benchmark.py"))

    if not steps:
        print("No steps selected. Exiting.")
        return 0

    print("Selected steps:")
    for s in steps:
        print(f"- {s}")

    try:
        for script_name in steps:
            _run_step(project_dir, script_name)
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed: {e}")
        return int(e.returncode) if e.returncode is not None else 1
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1

    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
