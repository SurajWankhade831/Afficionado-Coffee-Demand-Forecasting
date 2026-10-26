
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

METRICS_FILE = REPORTS_DIR / "demand_model_metrics.csv"
PREDICTIONS_FILE = REPORTS_DIR / "demand_model_predictions.csv"
IMPORTANCE_FILE = REPORTS_DIR / "demand_model_feature_importance.csv"

EVALUATION_FILE = REPORTS_DIR / "demand_model_evaluation.txt"
ERROR_SUMMARY_FILE = REPORTS_DIR / "demand_prediction_error_summary.csv"
ERROR_FIGURE = FIGURES_DIR / "demand_prediction_errors.png"


def main():
    """Summarize model metrics and prediction errors."""

    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}"
        )

    metrics = pd.read_csv(METRICS_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    required_columns = [
        "actual_quantity",
        "predicted_quantity",
        "absolute_error",
    ]

    missing = [
        column for column in required_columns
        if column not in predictions.columns
    ]

    if missing:
        raise ValueError(f"Prediction columns missing: {missing}")

    actual = predictions["actual_quantity"]
    predicted = predictions["predicted_quantity"]
    errors = actual - predicted
    absolute_errors = errors.abs()

    summary = pd.DataFrame(
        [
            {
                "test_rows": len(predictions),
                "mean_actual_quantity": actual.mean(),
                "mean_predicted_quantity": predicted.mean(),
                "mean_absolute_error": absolute_errors.mean(),
                "median_absolute_error": absolute_errors.median(),
                "maximum_absolute_error": absolute_errors.max(),
                "mean_signed_error": errors.mean(),
                "note": (
                    "Evaluation of held-out aggregated store-hour rows; "
                    "not a future-date forecast evaluation."
                ),
            }
        ]
    )

    summary.to_csv(ERROR_SUMMARY_FILE, index=False)

    metric_row = metrics.iloc[0]

    report_lines = [
        "DEMAND-PATTERN MODEL EVALUATION",
        "=" * 60,
        "",
        f"Model: {metric_row['model']}",
        f"Target: {metric_row['target']}",
        f"Test rows: {len(predictions)}",
        "",
        "MODEL METRICS",
        "-" * 60,
        f"MAE: {float(metric_row['MAE']):.3f}",
        f"RMSE: {float(metric_row['RMSE']):.3f}",
        f"R-squared: {float(metric_row['R2']):.3f}",
        "",
        "PREDICTION ERROR SUMMARY",
        "-" * 60,
        f"Mean actual quantity: {actual.mean():.3f}",
        f"Mean predicted quantity: {predicted.mean():.3f}",
        f"Median absolute error: {absolute_errors.median():.3f}",
        f"Maximum absolute error: {absolute_errors.max():.3f}",
        f"Mean signed error: {errors.mean():.3f}",
        "",
        "INTERPRETATION",
        "-" * 60,
        (
            "The model explains a limited share of the variation in the "
            "held-out store-hour observations. Its results should be "
            "treated as preliminary."
        ),
        (
            "The evaluation uses only a small random holdout of 11 "
            "aggregated store-hour rows."
        ),
        (
            "The source dataset has no complete transaction-date column. "
            "Therefore, this evaluation does not measure performance "
            "on future dates, days, or weeks."
        ),
        (
            "For dependable operational forecasting, obtain transaction "
            "dates and build a chronological dataset with enough history "
            "to support time-based validation."
        ),
        "",
        "FEATURE IMPORTANCE",
        "-" * 60,
    ]

    if IMPORTANCE_FILE.exists():
        importance = pd.read_csv(IMPORTANCE_FILE)

        for _, row in importance.head(10).iterrows():
            report_lines.append(
                f"{row['feature']}: {float(row['importance']):.4f}"
            )
    else:
        report_lines.append("Feature-importance file was not found.")

    EVALUATION_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    # Visualize the signed prediction errors.
    plt.figure(figsize=(9, 5))
    plt.bar(range(1, len(errors) + 1), errors)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Test observation")
    plt.ylabel("Actual minus predicted quantity")
    plt.title("Demand Model Prediction Errors")
    plt.tight_layout()
    plt.savefig(ERROR_FIGURE, dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT CREATED")
    print("=" * 60)
    print(f"Test observations: {len(predictions)}")
    print(f"Mean absolute error: {absolute_errors.mean():.3f}")
    print(f"Median absolute error: {absolute_errors.median():.3f}")
    print(f"Maximum absolute error: {absolute_errors.max():.3f}")

    print("\nCreated files:")
    print(EVALUATION_FILE)
    print(ERROR_SUMMARY_FILE)
    print(ERROR_FIGURE)

    print(
        "\nReminder: this evaluates held-out aggregated observations, "
        "not future calendar dates."
    )


if __name__ == "__main__":
    main()
