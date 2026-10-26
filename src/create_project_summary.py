
from pathlib import Path
import json

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

EDA_FILE = REPORTS_DIR / "eda_summary.json"
HOURLY_FILE = REPORTS_DIR / "hourly_demand_summary.csv"
MODEL_METRICS_FILE = REPORTS_DIR / "demand_model_metrics.csv"
ERROR_SUMMARY_FILE = REPORTS_DIR / "demand_prediction_error_summary.csv"
IMPORTANCE_FILE = REPORTS_DIR / "demand_model_feature_importance.csv"

OUTPUT_FILE = REPORTS_DIR / "final_project_summary.txt"


def main():
    """Create a consolidated summary from existing project outputs."""

    required_files = [
        EDA_FILE,
        HOURLY_FILE,
        MODEL_METRICS_FILE,
        ERROR_SUMMARY_FILE,
        IMPORTANCE_FILE,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "The following required files are missing:\n" + missing_text
        )

    with EDA_FILE.open("r", encoding="utf-8") as file:
        eda = json.load(file)

    hourly = pd.read_csv(HOURLY_FILE)
    metrics = pd.read_csv(MODEL_METRICS_FILE)
    errors = pd.read_csv(ERROR_SUMMARY_FILE)
    importance = pd.read_csv(IMPORTANCE_FILE)

    # Support the current report structure while avoiding invented values.
    def find_value(mapping, possible_keys, default=None):
        if isinstance(mapping, dict):
            for key in possible_keys:
                if key in mapping:
                    return mapping[key]

            for value in mapping.values():
                if isinstance(value, dict):
                    found = find_value(value, possible_keys, default=None)
                    if found is not None:
                        return found

        return default

    record_count = find_value(
        eda, ["row_count", "total_rows", "records", "dataset_rows"],
        default="See EDA report",
    )
    missing_count = find_value(
        eda, ["total_missing_values", "missing_values", "missing_count"],
        default="See EDA report",
    )
    duplicate_count = find_value(
        eda, ["duplicate_rows", "duplicate_count", "duplicates"],
        default="See EDA report",
    )
    total_quantity = find_value(
        eda, ["total_quantity", "total_quantity_sold", "quantity_sold"],
        default="See EDA report",
    )
    total_revenue = find_value(
        eda, ["calculated_revenue", "total_revenue", "revenue"],
        default="See EDA report",
    )

    # Find the peak hour from the hourly summary.
    hour_col = "transaction_hour"
    quantity_col = "total_quantity"
    revenue_col = "total_revenue"

    for col in [hour_col, quantity_col, revenue_col]:
        if col not in hourly.columns:
            raise ValueError(
                f"Expected column '{col}' was not found in {HOURLY_FILE.name}"
            )

    hourly[quantity_col] = pd.to_numeric(hourly[quantity_col], errors="coerce")
    hourly[revenue_col] = pd.to_numeric(hourly[revenue_col], errors="coerce")

    peak_quantity_row = hourly.loc[hourly[quantity_col].idxmax()]
    peak_revenue_row = hourly.loc[hourly[revenue_col].idxmax()]

    model = metrics.iloc[0]
    error = errors.iloc[0]

    lines = [
        "AFFICIONADO COFFEE ROASTERS",
        "DATA-DRIVEN FORECASTING & PEAK DEMAND PREDICTION",
        "CONSOLIDATED PROJECT SUMMARY",
        "=" * 68,
        "",
        "1. PROJECT OBJECTIVE",
        "-" * 68,
        (
            "Analyze historical transaction patterns, identify peak-demand "
            "hours, compare demand across stores, and evaluate a preliminary "
            "model for aggregated store-hour demand."
        ),
        "",
        "2. DATASET OVERVIEW",
        "-" * 68,
        f"Dataset records: {record_count}",
        f"Missing values: {missing_count}",
        f"Duplicate rows: {duplicate_count}",
        f"Total quantity sold: {total_quantity}",
        f"Calculated revenue: {total_revenue}",
        "",
        "3. PEAK-DEMAND FINDINGS",
        "-" * 68,
        (
            f"Peak hour by quantity in the hourly summary: "
            f"{int(peak_quantity_row[hour_col]):02d}:00 "
            f"({peak_quantity_row[quantity_col]:,.0f} units)."
        ),
        (
            f"Peak hour by revenue in the hourly summary: "
            f"{int(peak_revenue_row[hour_col]):02d}:00 "
            f"(${peak_revenue_row[revenue_col]:,.2f})."
        ),
        "",
        "4. PRELIMINARY MODEL RESULTS",
        "-" * 68,
        f"Model: {model['model']}",
        f"Target: {model['target']}",
        f"Test rows: {int(model['test_rows'])}",
        f"MAE: {float(model['MAE']):.3f}",
        f"RMSE: {float(model['RMSE']):.3f}",
        f"R-squared: {float(model['R2']):.3f}",
        f"Median absolute error: {float(error['median_absolute_error']):.3f}",
        f"Maximum absolute error: {float(error['maximum_absolute_error']):.3f}",
        "",
        "5. TOP MODEL FEATURES",
        "-" * 68,
    ]

    for _, row in importance.head(5).iterrows():
        lines.append(
            f"{row['feature']}: {float(row['importance']):.4f}"
        )

    lines.extend(
        [
            "",
            "6. DATA LIMITATIONS AND INTERPRETATION",
            "-" * 68,
            (
                "The source dataset contains time-of-day information but no "
                "complete transaction-date column."
            ),
            (
                "Consequently, this project cannot establish genuine daily, "
                "weekly, or multiweek forecasts, nor can it perform "
                "chronological validation across future dates."
            ),
            (
                "The Random Forest model was evaluated on a random holdout "
                "of aggregated store-hour rows. There are only 43 store-hour "
                "rows in the modeling dataset and 11 test rows."
            ),
            (
                "The current R-squared score is preliminary and does not "
                "demonstrate reliable future forecasting performance."
            ),
            (
                "For operational forecasting, obtain transaction dates and "
                "build a chronological dataset with sufficient history. "
                "Then compare baseline and time-series models using "
                "time-based validation."
            ),
            "",
            "7. GENERATED PROJECT OUTPUTS",
            "-" * 68,
            "Hourly demand summaries by store and category.",
            "Exploratory data analysis reports and charts.",
            "Model predictions, metrics, and feature importance.",
            "Actual-versus-predicted and prediction-error charts.",
            "Model evaluation report.",
            "",
            "END OF SUMMARY",
        ]
    )

    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("CONSOLIDATED PROJECT SUMMARY CREATED")
    print("=" * 60)
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Peak quantity hour: {int(peak_quantity_row[hour_col]):02d}:00")
    print(f"Peak revenue hour: {int(peak_revenue_row[hour_col]):02d}:00")
    print(f"Model R-squared: {float(model['R2']):.3f}")
    print("\nThe report includes the project's forecasting limitations.")


if __name__ == "__main__":
    main()
