
from pathlib import Path
import json

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

RAW_FILE = DATA_DIR / "Afficionado_Coffee_Raw.xlsx"

for folder in [FIGURES_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    """Load the designated raw dataset."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Expected dataset not found: {RAW_FILE}"
        )

    print(f"Loading dataset: {RAW_FILE.name}")

    df = pd.read_excel(RAW_FILE)

    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    return df


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    """Find a column using candidate names."""

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


def save_bar_chart(series, title, xlabel, ylabel, filename,
                   horizontal=False):
    """Save a bar chart without displaying a GUI window."""

    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    if horizontal:
        series.sort_values().plot(kind="barh", ax=ax)
    else:
        series.plot(kind="bar", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not horizontal:
        ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ANALYSIS
# ============================================================

def run_analysis():
    """Validate data and generate exploratory sales reports."""

    df = load_dataset()

    print("\n" + "=" * 65)
    print("AFFICIONADO COFFEE")
    print("HOURLY DEMAND AND PEAK-PERIOD ANALYSIS")
    print("=" * 65)

    print(f"\nDataset rows: {len(df):,}")
    print(f"Dataset columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())

    missing = df.isna().sum()

    missing_report = pd.DataFrame({
        "column": missing.index,
        "missing_values": missing.values,
        "missing_percent": (
            missing.values / max(len(df), 1) * 100
        ).round(2),
    })

    missing_report.to_csv(
        REPORTS_DIR / "missing_values_report.csv",
        index=False,
    )

    duplicate_rows = int(df.duplicated().sum())

    print(f"\nDuplicate rows: {duplicate_rows:,}")
    print(f"Total missing values: {int(missing.sum()):,}")

    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        numeric_df.describe().T.to_csv(
            REPORTS_DIR / "descriptive_statistics.csv"
        )

    # This dataset has year and time, but no full transaction date.
    date_columns = [
        column for column in df.columns
        if column in {
            "transaction_date",
            "date",
            "order_date",
            "sales_date",
            "datetime",
            "timestamp",
        }
    ]

    date_col = date_columns[0] if date_columns else None

    if date_col:
        parsed_dates = pd.to_datetime(
            df[date_col], errors="coerce"
        )
        valid_dates = parsed_dates.dropna()
        date_min = str(valid_dates.min()) if not valid_dates.empty else None
        date_max = str(valid_dates.max()) if not valid_dates.empty else None
        valid_date_count = int(len(valid_dates))
    else:
        date_min = None
        date_max = None
        valid_date_count = 0
        print(
            "\nDATA LIMITATION: No full transaction-date column exists."
        )
        print(
            "True daily and multiweek forecasting cannot be "
            "trained or validated with this dataset alone."
        )

    # Identify relevant columns.
    quantity_col = find_column(
        df, ["transaction_qty", "quantity", "qty", "units_sold"]
    )
    price_col = find_column(
        df, ["unit_price", "price", "selling_price"]
    )
    time_col = find_column(
        df, ["transaction_time", "time", "order_time"]
    )
    store_col = find_column(
        df, ["store_location", "store_id", "store"]
    )
    category_col = find_column(
        df, ["product_category", "category"]
    )

    total_quantity = None
    total_revenue = None
    peak_hour = None
    peak_hour_quantity = None
    peak_hour_revenue = None

    if quantity_col:
        df[quantity_col] = pd.to_numeric(
            df[quantity_col], errors="coerce"
        )

        total_quantity = float(df[quantity_col].sum())
        print(f"\nTotal quantity sold: {total_quantity:,.0f}")

    if price_col:
        df[price_col] = pd.to_numeric(
            df[price_col], errors="coerce"
        )

    if quantity_col and price_col:
        df["calculated_revenue"] = (
            df[quantity_col] * df[price_col]
        )
        total_revenue = float(df["calculated_revenue"].sum())
        print(f"Calculated revenue: ${total_revenue:,.2f}")

    # Revenue and quantity by store.
    if store_col and quantity_col:
        quantity_by_store = (
            df.groupby(store_col)[quantity_col]
            .sum()
            .sort_values(ascending=False)
        )
        quantity_by_store.to_csv(
            REPORTS_DIR / "quantity_by_store.csv",
            header=["total_quantity"],
        )
        save_bar_chart(
            quantity_by_store,
            "Quantity Sold by Store",
            "Quantity",
            "Store",
            "quantity_by_store.png",
            horizontal=True,
        )

    if store_col and "calculated_revenue" in df.columns:
        revenue_by_store = (
            df.groupby(store_col)["calculated_revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        revenue_by_store.to_csv(
            REPORTS_DIR / "revenue_by_store.csv",
            header=["total_revenue"],
        )
        save_bar_chart(
            revenue_by_store,
            "Revenue by Store",
            "Revenue ($)",
            "Store",
            "revenue_by_store.png",
            horizontal=True,
        )

    # Revenue and quantity by category.
    if category_col and quantity_col:
        quantity_by_category = (
            df.groupby(category_col)[quantity_col]
            .sum()
            .sort_values(ascending=False)
        )
        quantity_by_category.to_csv(
            REPORTS_DIR / "quantity_by_category.csv",
            header=["total_quantity"],
        )
        save_bar_chart(
            quantity_by_category,
            "Quantity Sold by Product Category",
            "Quantity",
            "Category",
            "quantity_by_category.png",
            horizontal=True,
        )

    if category_col and "calculated_revenue" in df.columns:
        revenue_by_category = (
            df.groupby(category_col)["calculated_revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        revenue_by_category.to_csv(
            REPORTS_DIR / "revenue_by_category.csv",
            header=["total_revenue"],
        )
        save_bar_chart(
            revenue_by_category,
            "Revenue by Product Category",
            "Revenue ($)",
            "Category",
            "revenue_by_category.png",
            horizontal=True,
        )

    # Hourly sales and demand.
    hourly_df = pd.DataFrame()

    if time_col:
        parsed_time = pd.to_datetime(
            df[time_col].astype(str),
            format="mixed",
            errors="coerce",
        )

        valid_time = parsed_time.notna()
        hourly_df = df.loc[valid_time].copy()
        hourly_df["transaction_hour"] = (
            parsed_time.loc[valid_time].dt.hour
        )

        aggregations = {
            "transaction_count": ("transaction_hour", "size")
        }

        if quantity_col:
            aggregations["total_quantity"] = (
                quantity_col, "sum"
            )

        if "calculated_revenue" in hourly_df.columns:
            aggregations["total_revenue"] = (
                "calculated_revenue", "sum"
            )

        hourly_summary = (
            hourly_df.groupby("transaction_hour")
            .agg(**aggregations)
            .sort_index()
        )

        hourly_summary.to_csv(
            REPORTS_DIR / "hourly_demand_summary.csv"
        )

        if "total_quantity" in hourly_summary.columns:
            save_bar_chart(
                hourly_summary["total_quantity"],
                "Demand by Hour of Day",
                "Hour (24-hour clock)",
                "Units Sold",
                "quantity_by_hour.png",
            )

            peak_hour = int(
                hourly_summary["total_quantity"].idxmax()
            )
            peak_hour_quantity = float(
                hourly_summary["total_quantity"].max()
            )

        if "total_revenue" in hourly_summary.columns:
            save_bar_chart(
                hourly_summary["total_revenue"],
                "Revenue by Hour of Day",
                "Hour (24-hour clock)",
                "Revenue ($)",
                "revenue_by_hour.png",
            )

            peak_hour_revenue = int(
                hourly_summary["total_revenue"].idxmax()
            )

        if peak_hour is not None:
            print(
                f"\nPeak hour by quantity: {peak_hour:02d}:00"
            )
            print(
                f"Quantity during peak hour: "
                f"{peak_hour_quantity:,.0f}"
            )

        if peak_hour_revenue is not None:
            print(
                f"Peak revenue hour: {peak_hour_revenue:02d}:00"
            )

        # Hourly demand by store, when both columns are available.
        if store_col and quantity_col:
            store_hour = (
                hourly_df.groupby(
                    [store_col, "transaction_hour"]
                )[quantity_col]
                .sum()
                .unstack(fill_value=0)
            )

            store_hour.to_csv(
                REPORTS_DIR / "hourly_quantity_by_store.csv"
            )

            fig, ax = plt.subplots(figsize=(12, 6))
            image = ax.imshow(
                store_hour.values,
                aspect="auto",
                interpolation="nearest",
            )

            ax.set_title("Hourly Quantity by Store")
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Store")
            ax.set_xticks(range(len(store_hour.columns)))
            ax.set_xticklabels(store_hour.columns)
            ax.set_yticks(range(len(store_hour.index)))
            ax.set_yticklabels(store_hour.index)

            fig.colorbar(image, ax=ax, label="Units Sold")
            fig.tight_layout()
            fig.savefig(
                FIGURES_DIR / "hourly_demand_heatmap.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    # Summary report.
    summary = {
        "project": (
            "Data-Driven Forecasting and Peak Demand "
            "Prediction for Afficionado Coffee Roasters"
        ),
        "dataset_file": RAW_FILE.name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "column_names": list(df.columns),
        "duplicate_rows": duplicate_rows,
        "missing_values_total": int(df.isna().sum().sum()),
        "date_column_detected": date_col,
        "valid_date_count": valid_date_count,
        "earliest_date": date_min,
        "latest_date": date_max,
        "quantity_column": quantity_col,
        "price_column": price_col,
        "total_quantity": total_quantity,
        "calculated_revenue": total_revenue,
        "peak_quantity_hour": peak_hour,
        "peak_hour_quantity": peak_hour_quantity,
        "peak_revenue_hour": peak_hour_revenue,
        "daily_forecasting_ready": bool(
            date_col is not None and valid_date_count > 0
        ),
        "hourly_pattern_analysis_available": bool(time_col),
        "forecasting_limitation": (
            "No full transaction date. Historical daily and "
            "multiweek forecasting and time-based validation "
            "are not supported by the supplied dataset."
        ),
    }

    summary_path = REPORTS_DIR / "eda_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("ANALYSIS FINISHED")
    print("=" * 65)
    print(f"Summary: {summary_path}")
    print(f"Reports: {REPORTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    run_analysis()
