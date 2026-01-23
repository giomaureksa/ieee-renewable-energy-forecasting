# Results & Model Evaluation

This document summarizes the performance of the energy demand forecasting
models developed in this project. The evaluation focuses on both validation
and final test periods, using industry-standard regression metrics.

---

## 1. Evaluation Setup

### Forecast Windows
- **Training period:** Jan 2016 – Sep 2020
- **Validation period:** Oct 2020
- **Test period:** Nov 2020

### Target Variable
- `value` — electrical energy demand (time series)

### Evaluation Metrics
The following metrics are used:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score (Coefficient of Determination)**

---

## 2. Validation Performance (October 2020)

| Model     | RMSE | MAE | R² |
|----------|------|-----|----|
| LightGBM | 11.25 | 2.46 | 0.991 |
| XGBoost  | **6.15** | **1.84** | **0.997** |

**Observations:**
- Both models achieved strong predictive performance.
- XGBoost consistently outperformed LightGBM across all metrics.
- Low MAE indicates very small average deviation from actual demand values.

---

## 3. Test Performance (November 2020)

| Model     | RMSE | MAE | R² |
|----------|------|-----|----|
| LightGBM | 18.38 | 4.34 | 0.983 |
| XGBoost  | **16.95** | **3.33** | **0.986** |

**Observations:**
- Performance degradation from validation to test is expected and normal.
- XGBoost remained more robust under unseen data conditions.
- R² scores above 0.98 indicate strong explanatory power.

---

## 4. Error Interpretation (Contextualized)

Given that typical energy demand values are in the range of **~1000–1500 units**:

- **MAE ~3–4** 
  → Average prediction error is **less than 0.5%** of actual demand.
- **RMSE ~17** 
  → Larger errors are rare but penalized more heavily.

These values indicate **high-quality forecasts suitable for operational use**.

---

## 5. Model Selection

Based on validation and test results:

**Final Selected Model:** **XGBoost**

**Rationale:**
- Lower RMSE and MAE
- Better generalization on unseen data
- More stable performance across time windows

LightGBM is retained as a strong baseline and comparison model.

---

## 6. Saved Outputs

The following artifacts are stored in the `results/` directory:


