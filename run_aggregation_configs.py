import os
import subprocess
from pathlib import Path

AGGREGATION_STRATEGIES = [
    {"name": "equal_avg", "weights": [0.25, 0.25, 0.25, 0.25]},
    {"name": "bias_best_light", "weights": [0.4, 0.3, 0.2, 0.1]},
    {"name": "bias_best_heavy", "weights": [0.7, 0.1, 0.1, 0.1]},
    {"name": "diagonal_balance", "weights": [0.4, 0.1, 0.1, 0.4]},
    {"name": "split_cluster_A", "weights": [0.5, 0.5, 0.0, 0.0]},
    {"name": "split_cluster_B", "weights": [0.0, 0.0, 0.5, 0.5]},
    {"name": "drop_model_1", "weights": [0.0, 0.33, 0.33, 0.34]},
    {"name": "drop_model_4", "weights": [0.34, 0.33, 0.33, 0.0]},
    {"name": "bad_run_noise", "weights": [0.99, 0.01, 0.0, 0.0]},
    {"name": "poisson_weights", "weights": [0.537, 0.268, 0.134, 0.061]},
]

CHECKPOINTS = [
    "checkpoints/wandering-river-4.pt",
    "checkpoints/smart-smoke-5.pt",
    "checkpoints/hearty-bee-7.pt",
    "checkpoints/eager-haze-6.pt",
]
BASE_SAVE_DIR = Path("whisper-aggregated")
EVALUATION_DATASET = "i4ds/spc_r"
EVALUATION_CONFIG = None
EVALUATION_SPLIT = "test"


def run_aggregation_and_eval():
    BASE_SAVE_DIR.mkdir(exist_ok=True)

    # Sanity check: do checkpoints exist?
    missing_checkpoints = [c for c in CHECKPOINTS if not Path(c).exists()]
    if missing_checkpoints:
        print("WARNING: The following checkpoints are missing:")
        for c in missing_checkpoints:
            print(f"   {c}")
        print("The script will skip actual aggregation/evaluation for now.\n")

    for strategy in AGGREGATION_STRATEGIES:
        name = strategy["name"]
        weights = strategy["weights"]
        print(f"\nRunning strategy: {name} with weights {weights}")

        # Create a subdir for this strategy
        save_dir = BASE_SAVE_DIR / name
        save_dir.mkdir(exist_ok=True)

        # Aggregate command
        aggregate_cmd = [
            "python",
            "average_runner.py",
            "--checkpoints",
            *CHECKPOINTS,
            "--weights",
            *[str(w) for w in weights],
            "--save_dir",
            str(save_dir),
        ]

        if missing_checkpoints:
            print(
                f"[SANITY CHECK] Would run aggregation command:\n{' '.join(aggregate_cmd)}"
            )
        else:
            print(f"Running aggregation command:\n{' '.join(aggregate_cmd)}")
            subprocess.run(aggregate_cmd, check=True)

        # Evaluation command
        examples_csv = save_dir / "evaluation_examples.csv"
        metrics_csv = save_dir / "evaluation_metrics.csv"
        eval_cmd = [
            "python",
            "evaluate_model.py",
            "--model_dir",
            str(save_dir),
            "--dataset_name",
            EVALUATION_DATASET,
            "--split",
            EVALUATION_SPLIT,
            "--precision",
            "float16",
            "--examples_csv",
            str(examples_csv),
            "--metrics_csv",
            str(metrics_csv),
        ]

        if EVALUATION_CONFIG:
            eval_cmd.extend(["--dataset_config", EVALUATION_CONFIG])

        if missing_checkpoints:
            print(f"[SANITY CHECK] Would run evaluation command:\n{' '.join(eval_cmd)}")
        else:
            print(f"Running evaluation command:\n{' '.join(eval_cmd)}")
            subprocess.run(eval_cmd, check=True)

        print(f"   Completed strategy: {name}")
        print(f"   Aggregated model: {save_dir}")
        print(f"   Examples CSV: {examples_csv}")
        print(f"   Metrics CSV: {metrics_csv}")


if __name__ == "__main__":
    run_aggregation_and_eval()
