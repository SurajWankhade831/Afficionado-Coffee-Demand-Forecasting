# Data-Driven Forecasting & Peak Demand Prediction for Afficionado Coffee Roasters

## 1. Project Overview

This project analyzes transaction data from Afficionado Coffee Roasters to identify hourly demand patterns, determine peak business hours, compare demand across stores and product categories, and evaluate a machine-learning model for estimating aggregated store-hour demand.

The project uses Python, pandas, Matplotlib, and scikit-learn for data preparation, exploratory analysis, visualization, modeling, and evaluation.

**Important scope note:** The available dataset contains transaction times but no complete transaction-date column. Therefore, this version focuses on historical time-of-day demand patterns and preliminary demand estimation. It does not claim to provide validated forecasts for future dates or weeks.

## 2. Project Objectives

- Inspect dataset quality and summarize transaction data.
- Analyze hourly sales quantity and calculated revenue.
- Identify peak-demand hours.
- Compare demand across stores and product categories.
- Train a Random Forest regression model on aggregated store-hour observations.
- Evaluate model errors and document limitations.
- Produce reports and charts to support business planning.

## 3. Dataset

The project uses the Afficionado Coffee Roasters transaction dataset supplied for the assignment.

The available columns include:

- `transaction_id`
- `year`
- `transaction_time`
- `transaction_qty`
- `store_id`
- `store_location`
- `product_id`
- `unit_price`
- `product_category`
- `product_type`
- `product_detail`

The analysis calculates revenue as:

`Revenue = transaction_qty × unit_price`

### Dataset summary

- Transaction records: 149,116
- Total units sold: 214,470
- Calculated revenue: $698,812.33
- Duplicate rows: 0
- Missing values: 0

These figures describe the supplied dataset and should not be interpreted as independently verified business accounts.

## 4. Key Findings

- **Peak recorded hour by quantity:** 10:00 AM
- **Peak recorded hour by revenue:** 10:00 AM
- **Strong demand period:** Morning hours, especially 8:00–10:00 AM.
- **Lowest recorded hourly activity:** 8:00 PM, based on the supplied time-of-day records.

These are aggregated time-of-day findings, not evidence of demand on a particular date.

## 5. Machine-Learning Model

A Random Forest regression model was trained to estimate aggregated quantity by store and hour.

The current evaluation used a random holdout of aggregated store-hour rows.

| Metric | Result |
|---|---:|
| Training rows | 32 |
| Test rows | 11 |
| Mean Absolute Error (MAE) | 682.654 |
| Root Mean Squared Error (RMSE) | 1,080.529 |
| R-squared | 0.263 |

### Interpretation

The model explains approximately 26.3% of the variation in the held-out observations in this particular split. The test set is very small, and the errors are substantial. Results are preliminary and should not be treated as evidence of reliable future forecasting performance.

## 6. Data Limitation and Future Improvements

The dataset does not contain a complete transaction date for each transaction. Consequently, it cannot support genuine daily or multiweek time-series forecasting or chronological validation across future dates.

Recommended improvements:

1. Obtain transaction-level records containing actual dates and times.
2. Build a chronological hourly or daily demand series.
3. Investigate missing time intervals and whether the data covers consecutive dates.
4. Establish time-based training, validation, and test periods.
5. Compare seasonal-naive baselines with suitable forecasting models.
6. Evaluate forecasts using MAE, RMSE, and appropriate business metrics.
7. Investigate store-level and category-level patterns when sufficient dated data is available.

## 7. Technology Stack

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- statsmodels
- openpyxl
- Streamlit and Plotly (available in the project environment for potential interactive reporting)
- joblib

## 8. Project Structure

```text
Afficionado_Coffee_Forecasting/
├── data/
├── models/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── reports/
├── research_paper/
├── src/
│   ├── analyze_data.py
│   ├── prepare_demand_data.py
│   ├── model_demand_patterns.py
│   ├── evaluate_demand_model.py
│   └── create_project_summary.py
├── requirements.txt
└── README.md
```

## 9. Setup Instructions

Open PowerShell in the project directory.

Create a virtual environment if one does not already exist:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Ensure the source workbook is located at:

`data/Afficionado_Coffee_Raw.xlsx`

## 10. Running the Project

Run the scripts from the project root in this order:

```powershell
python src/analyze_data.py
python src/prepare_demand_data.py
python src/model_demand_patterns.py
python src/evaluate_demand_model.py
python src/create_project_summary.py
```

The scripts write reports to `outputs/reports/` and visualizations to `outputs/figures/`.

## 11. Main Outputs

The project generates:

- Dataset quality and exploratory-analysis reports
- Hourly demand summaries by store and category
- Hourly demand and revenue charts
- Demand-model predictions and performance metrics
- Feature-importance results
- Actual-versus-predicted chart
- Prediction-error chart
- Model evaluation report
- Consolidated project summary

## 12. Conclusion

The analysis identifies 10:00 AM as the peak recorded hour for both quantity and calculated revenue. The machine-learning model provides a preliminary assessment of aggregated store-hour demand, but its current performance and the absence of transaction dates limit its forecasting value.

Obtaining dated transaction data and validating models chronologically are essential next steps before using this work for operational forecasting or staffing and inventory decisions.