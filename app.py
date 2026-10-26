
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Afficionado Coffee | Demand Analytics",
    page_icon="☕",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"

st.title("☕ Afficionado Coffee Roasters")
st.subheader("Data-Driven Forecasting & Peak Demand Prediction")
st.caption("Hourly demand analysis • Store performance • Preliminary demand modelling")

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_csv(filename):
    path = REPORTS / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_summary():
    path = REPORTS / "eda_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


summary = load_summary()
hourly = load_csv("hourly_demand_summary.csv")
store_demand = load_csv("quantity_by_store.csv")
category_demand = load_csv("quantity_by_category.csv")
hourly_store = load_csv("hourly_demand_by_store.csv")
hourly_category = load_csv("hourly_demand_by_category.csv")
revenue_hour = load_csv("revenue_by_hour.csv")
revenue_store = load_csv("revenue_by_store.csv")
revenue_category = load_csv("revenue_by_category.csv")
metrics = load_csv("demand_model_metrics.csv")
importance = load_csv("demand_model_feature_importance.csv")
predictions = load_csv("demand_model_predictions.csv")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "Project Overview",
        "Peak Demand Analysis",
        "Store Performance",
        "Product Categories",
        "Model Evaluation",
        "Project Limitations",
    ],
)

st.sidebar.info(
    "This dashboard analyses recorded time-of-day patterns. "
    "The source dataset does not contain full transaction dates."
)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def metric_card(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)


def show_chart(data, x, y, title, chart_type="bar", color=None):
    if data.empty or x not in data.columns or y not in data.columns:
        st.info(f"Chart unavailable: required data for '{title}' was not found.")
        return

    if chart_type == "line":
        fig = px.line(data, x=x, y=y, title=title, markers=True, color=color)
    else:
        fig = px.bar(data, x=x, y=y, title=title, color=color)

    fig.update_layout(
        template="plotly_white",
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_existing_figure(filename, caption):
    path = FIGURES / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def find_column(data, candidates):
    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    return None


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------
if page == "Project Overview":
    st.header("Project Overview")

    st.write(
        "This project examines historical coffee-shop demand patterns, "
        "identifies peak transaction hours, compares store and product-category "
        "performance, and evaluates a preliminary machine-learning model."
    )

    c1, c2, c3, c4 = st.columns(4)

    records = summary.get("total_records", summary.get("rows", 149116))
    quantity = summary.get("total_quantity", summary.get("total_units_sold", 214470))
    revenue = summary.get("total_revenue", 698812.33)
    peak_hour = summary.get("peak_hour", "10:00")

    with c1:
        metric_card("Transactions", f"{int(records):,}")
    with c2:
        metric_card("Units Sold", f"{int(quantity):,}")
    with c3:
        try:
            metric_card("Calculated Revenue", f"${float(revenue):,.2f}")
        except (TypeError, ValueError):
            metric_card("Calculated Revenue", "See reports")
    with c4:
        metric_card("Peak Recorded Hour", str(peak_hour))

    st.divider()
    st.subheader("Hourly Demand Pattern")
    show_existing_figure(
        "quantity_by_hour.png",
        "Recorded quantity sold by transaction hour",
    )

    st.subheader("Key Findings")
    st.markdown(
        """
        - Recorded demand is strongest around the morning peak.
        - 10:00 is the peak recorded hour by quantity in the current analysis.
        - Store and product-category demand can help inform staffing and stock planning.
        - The model is preliminary and does not establish future-date forecasting accuracy.
        """
    )

# --------------------------------------------------
# PAGE 2: PEAK DEMAND
# --------------------------------------------------
elif page == "Peak Demand Analysis":
    st.header("Peak Demand Analysis")
    st.write(
        "Explore recorded demand by hour. These are time-of-day aggregates, "
        "not a continuous timeline of consecutive calendar hours."
    )

    if not hourly.empty:
        hour_col = find_column(hourly, ["hour", "transaction_hour", "time", "transaction_time"])
        quantity_col = find_column(hourly, ["total_quantity", "quantity", "transaction_qty", "total_qty"])
        if hour_col and quantity_col:
            show_chart(hourly, hour_col, quantity_col, "Recorded Quantity by Hour")
        else:
            st.dataframe(hourly, use_container_width=True)
    else:
        show_existing_figure("quantity_by_hour.png", "Recorded quantity by hour")

    st.subheader("Hourly Demand Heatmap")
    show_existing_figure(
        "hourly_demand_heatmap.png",
        "Recorded demand distribution by store and hour",
    )

    if not hourly_category.empty:
        st.subheader("Demand by Hour and Category")
        hour_col = find_column(hourly_category, ["hour", "transaction_hour"])
        quantity_col = find_column(
            hourly_category,
            ["total_quantity", "quantity", "transaction_qty", "total_qty"],
        )
        category_col = find_column(
            hourly_category,
            ["product_category", "category"],
        )
        if hour_col and quantity_col and category_col:
            show_chart(
                hourly_category,
                hour_col,
                quantity_col,
                "Recorded Quantity by Hour and Category",
                color=category_col,
            )
        else:
            st.dataframe(hourly_category, use_container_width=True)

# --------------------------------------------------
# PAGE 3: STORE PERFORMANCE
# --------------------------------------------------
elif page == "Store Performance":
    st.header("Store Performance")

    if not store_demand.empty:
        store_col = find_column(store_demand, ["store_location", "store_id", "store"])
        quantity_col = find_column(
            store_demand,
            ["total_quantity", "quantity", "transaction_qty", "total_qty"],
        )
        if store_col and quantity_col:
            show_chart(
                store_demand,
                store_col,
                quantity_col,
                "Total Recorded Quantity by Store",
            )
        else:
            st.dataframe(store_demand, use_container_width=True)
    else:
        show_existing_figure("quantity_by_store.png", "Quantity sold by store")

    if not revenue_store.empty:
        st.subheader("Revenue by Store")
        store_col = find_column(revenue_store, ["store_location", "store_id", "store"])
        revenue_col = find_column(
            revenue_store,
            ["total_revenue", "revenue", "unit_price"],
        )
        if store_col and revenue_col:
            show_chart(revenue_store, store_col, revenue_col, "Calculated Revenue by Store")
        else:
            st.dataframe(revenue_store, use_container_width=True)

    st.subheader("Store-Level Hourly Demand")
    if not hourly_store.empty:
        st.dataframe(hourly_store, use_container_width=True)
    else:
        st.info("The hourly store report is not available.")

# --------------------------------------------------
# PAGE 4: PRODUCT CATEGORIES
# --------------------------------------------------
elif page == "Product Categories":
    st.header("Product Category Analysis")

    if not category_demand.empty:
        category_col = find_column(category_demand, ["product_category", "category"])
        quantity_col = find_column(
            category_demand,
            ["total_quantity", "quantity", "transaction_qty", "total_qty"],
        )
        if category_col and quantity_col:
            show_chart(
                category_demand,
                category_col,
                quantity_col,
                "Recorded Quantity by Product Category",
            )
        else:
            st.dataframe(category_demand, use_container_width=True)
    else:
        show_existing_figure("quantity_by_category.png", "Quantity by product category")

    if not revenue_category.empty:
        st.subheader("Calculated Revenue by Category")
        category_col = find_column(revenue_category, ["product_category", "category"])
        revenue_col = find_column(
            revenue_category,
            ["total_revenue", "revenue", "unit_price"],
        )
        if category_col and revenue_col:
            show_chart(
                revenue_category,
                category_col,
                revenue_col,
                "Calculated Revenue by Product Category",
            )
        else:
            st.dataframe(revenue_category, use_container_width=True)

    if not hourly_category.empty:
        st.subheader("Hourly Category Report")
        st.dataframe(hourly_category, use_container_width=True)

# --------------------------------------------------
# PAGE 5: MODEL EVALUATION
# --------------------------------------------------
elif page == "Model Evaluation":
    st.header("Preliminary Demand-Pattern Model")
    st.warning(
        "This model predicts aggregated store-hour demand patterns. "
        "It has not been validated for future calendar dates."
    )

    show_existing_figure(
        "actual_vs_predicted_demand.png",
        "Actual versus predicted demand on the held-out test sample",
    )
    show_existing_figure(
        "demand_prediction_errors.png",
        "Prediction error distribution",
    )

    if not metrics.empty:
        st.subheader("Model Metrics")
        st.dataframe(metrics, use_container_width=True)
    else:
        st.info("The model metrics report was not found.")

    if not importance.empty:
        st.subheader("Feature Importance")
        st.dataframe(importance, use_container_width=True)

    if not predictions.empty:
        st.subheader("Test-Sample Predictions")
        st.dataframe(predictions, use_container_width=True)

    st.caption(
        "The current evaluation uses a small random holdout of aggregated store-hour rows. "
        "Its metrics should not be interpreted as evidence of future-date forecasting performance."
    )

# --------------------------------------------------
# PAGE 6: LIMITATIONS
# --------------------------------------------------
elif page == "Project Limitations":
    st.header("Dataset Limitations and Responsible Interpretation")

    st.warning(
        "The supplied dataset has a transaction time but no full transaction date."
    )

    st.markdown(
        """
        **What the data supports**
        - Historical transaction and revenue summaries.
        - Comparison of recorded demand by hour, store, and product category.
        - Preliminary modelling of aggregated store-hour demand patterns.

        **What the data does not support on its own**
        - Genuine daily, weekly, or monthly demand forecasts.
        - Reliable forecasts for a particular future calendar date.
        - Calendar-based lag features such as the previous 24 or 168 hours.
        - Time-based validation across successive dates.

        **Recommended next step**

        Obtain transaction-level data with a complete date and time for each record.
        With that data, a proper time-series forecasting workflow can be built and
        validated using chronological train/test splits.
        """
    )

    st.subheader("Data Quality Summary")
    st.json(summary if summary else {"message": "EDA summary JSON was not found."})

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Afficionado Coffee Roasters | Academic analytics project | "
    "Interpret results within the documented dataset limitations."
)