
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"

RAW_FILE = DATA_DIR / "Afficionado_Coffee_Raw.xlsx"

OUTPUT_FILE = REPORTS_DIR / "hourly_demand_by_store.csv"
CATEGORY_FILE = REPORTS_DIR / "hourly_demand_by_category.csv"


# ============================================================
# DATA PREPARATION
# ============================================================

def main():
    """Aggregate transactions into hourly demand datasets."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_FILE}"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {RAW_FILE.name}")
    df = pd.read_excel(RAW_FILE)

    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    required_columns = [
        "transaction_time",
        "transaction_qty",
        "unit_price",
        "store_id",
        "store_location",
        "product_category",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    # Convert numeric fields safely.
    df["transaction_qty"] = pd.to_numeric(
        df["transaction_qty"], errors="coerce"
    )
    df["unit_price"] = pd.to_numeric(
        df["unit_price"], errors="coerce"
    )

    # Parse the time of day without inventing transaction dates.
    parsed_time = pd.to_datetime(
        df["transaction_time"].astype(str),
        format="mixed",
        errors="coerce",
    )

    valid = (
        parsed_time.notna()
        & df["transaction_qty"].notna()
        & df["unit_price"].notna()
    )

    df = df.loc[valid].copy()
    parsed_time = parsed_time.loc[valid]

    df["transaction_hour"] = parsed_time.dt.hour
    df["calculated_revenue"] = (
        df["transaction_qty"] * df["unit_price"]
    )

    # Create a store label for reporting.
    df["store_label"] = (
        df["store_location"].astype(str)
        + " (ID "
        + df["store_id"].astype(str)
        + ")"
    )

    # --------------------------------------------------------
    # Hourly demand by store
    # --------------------------------------------------------

    hourly_store = (
        df.groupby(
            ["store_id", "store_location", "transaction_hour"],
            as_index=False,
        )
        .agg(
            transaction_count=("transaction_id", "count"),
            total_quantity=("transaction_qty", "sum"),
            total_revenue=("calculated_revenue", "sum"),
        )
        .sort_values(
            ["store_id", "transaction_hour"]
        )
    )

    hourly_store.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------------
    # Hourly demand by store and product category
    # --------------------------------------------------------

    hourly_category = (
        df.groupby(
            [
                "store_id",
                "store_location",
                "transaction_hour",
                "product_category",
            ],
            as_index=False,
        )
        .agg(
            transaction_count=("transaction_id", "count"),
            total_quantity=("transaction_qty", "sum"),
            total_revenue=("calculated_revenue", "sum"),
        )
        .sort_values(
            [
                "store_id",
                "transaction_hour",
                "product_category",
            ]
        )
    )

    hourly_category.to_csv(CATEGORY_FILE, index=False)

    # --------------------------------------------------------
    # Summary and limitation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("HOURLY DEMAND DATA PREPARATION COMPLETE")
    print("=" * 60)

    print(f"Valid transaction records: {len(df):,}")
    print(f"Store-hour rows: {len(hourly_store):,}")
    print(f"Store-hour-category rows: {len(hourly_category):,}")

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Saved: {CATEGORY_FILE}")

    peak = (
        hourly_store.groupby("transaction_hour")["total_quantity"]
        .sum()
        .idxmax()
    )

    print(f"\nPeak recorded hour overall: {int(peak):02d}:00")

    print(
        "\nIMPORTANT: These files describe aggregated time-of-day "
        "patterns. They do not represent consecutive hourly "
        "observations across calendar dates. Do not use them as "
        "genuine daily or multiweek forecasting sequences."
    )


if __name__ == "__main__":
    main()
