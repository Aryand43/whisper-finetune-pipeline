from pathlib import Path

from average_runner import aggregate_models
from evaluate_model import evaluate_model

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

        if missing_checkpoints:
            print("[SANITY CHECK] Would average checkpoints via aggregate_models()")
        else:
            print("Aggregating checkpoints via aggregate_models()")
            aggregate_models(
                checkpoints=CHECKPOINTS,
                weights=weights,
                save_dir=str(save_dir),
                log_to_wandb=False,
            )

        examples_csv = save_dir / "evaluation_examples.csv"
        metrics_csv = save_dir / "evaluation_metrics.csv"

        if missing_checkpoints:
            print("[SANITY CHECK] Would run evaluation via evaluate_model()")
        else:
            print("Evaluating aggregated model via evaluate_model()")
            metrics = evaluate_model(
                model_dir=str(save_dir),
                dataset_name=EVALUATION_DATASET,
                split=EVALUATION_SPLIT,
                dataset_config=EVALUATION_CONFIG,
                precision="float16",
                examples_csv=str(examples_csv),
                metrics_csv=str(metrics_csv),
            )
            print(f"   Metrics: {metrics}")
            print(f"   Examples CSV: {examples_csv}")
            print(f"   Metrics CSV: {metrics_csv}")

        print(f"   Completed strategy: {name}")
        print(f"   Aggregated model: {save_dir}")


if __name__ == "__main__":
    run_aggregation_and_eval()
