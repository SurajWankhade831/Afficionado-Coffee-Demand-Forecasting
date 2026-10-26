
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

INPUT_FILE = REPORTS_DIR / "hourly_demand_by_store.csv"
CATEGORY_INPUT_FILE = REPORTS_DIR / "hourly_demand_by_category.csv"

PREDICTIONS_FILE = REPORTS_DIR / "demand_model_predictions.csv"
METRICS_FILE = REPORTS_DIR / "demand_model_metrics.csv"
IMPORTANCE_FILE = REPORTS_DIR / "demand_model_feature_importance.csv"
FIGURE_FILE = FIGURES_DIR / "actual_vs_predicted_demand.png"


def main():
    """Model hourly demand patterns without inventing calendar dates."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "store_id",
        "store_location",
        "transaction_hour",
        "transaction_count",
        "total_quantity",
        "total_revenue",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure the target and hour are numeric.
    df["transaction_hour"] = pd.to_numeric(
        df["transaction_hour"], errors="coerce"
    )
    df["total_quantity"] = pd.to_numeric(
        df["total_quantity"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "store_id",
            "store_location",
            "transaction_hour",
            "total_quantity",
        ]
    ).copy()

    # Store and hour are predictors; quantity is the target.
    X = df[["store_id", "store_location", "transaction_hour"]].copy()
    y = df["total_quantity"].astype(float)

    X["store_id"] = X["store_id"].astype(str)
    X["store_location"] = X["store_location"].astype(str)
    X["transaction_hour"] = X["transaction_hour"].astype(int)

    categorical_features = ["store_id", "store_location"]
    numeric_features = ["transaction_hour"]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # This evaluates how well the model fits unseen store-hour rows.
    # It does NOT validate future dates because dates are unavailable.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = pd.DataFrame(
        [
            {
                "model": "RandomForestRegressor",
                "target": "total_quantity",
                "test_rows": len(y_test),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "validation_type": "Random holdout of aggregated store-hour rows",
                "limitation": (
                    "Not a future-date forecast; dataset has no full "
                    "transaction dates."
                ),
            }
        ]
    )
    metrics.to_csv(METRICS_FILE, index=False)

    prediction_output = X_test.copy()
    prediction_output["actual_quantity"] = y_test.to_numpy()
    prediction_output["predicted_quantity"] = predictions
    prediction_output["absolute_error"] = (
        prediction_output["actual_quantity"]
        - prediction_output["predicted_quantity"]
    ).abs()
    prediction_output.to_csv(PREDICTIONS_FILE, index=False)

    # Save feature importance from the fitted Random Forest.
    fitted_preprocessor = model.named_steps["preprocessing"]
    fitted_regressor = model.named_steps["regressor"]
    feature_names = fitted_preprocessor.get_feature_names_out()

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": fitted_regressor.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv(IMPORTANCE_FILE, index=False)

    # Create an actual-vs-predicted diagnostic figure.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.8)
    maximum = max(float(y_test.max()), float(predictions.max()))
    plt.plot([0, maximum], [0, maximum], linestyle="--")
    plt.xlabel("Actual hourly quantity")
    plt.ylabel("Predicted hourly quantity")
    plt.title("Actual vs Predicted Demand by Store and Hour")
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("DEMAND-PATTERN MODELING COMPLETE")
    print("=" * 60)
    print(f"Rows used: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R-squared (R2): {r2:.3f}")

    print("\nSaved reports:")
    print(METRICS_FILE)
    print(PREDICTIONS_FILE)
    print(IMPORTANCE_FILE)
    print(FIGURE_FILE)

    print(
        "\nIMPORTANT: These metrics measure performance on held-out "
        "aggregated store-hour rows. They do not establish accuracy "
        "for future dates or future weeks."
    )


if __name__ == "__main__":
    main()
