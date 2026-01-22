# Energy Demand Forecasting Using Gradient Boosting Models

## Overview
This project demonstrates a **production-style machine learning pipeline**
for energy demand forecasting using gradient boosting models.
The pipeline is designed to reflect real-world industrial workflows,
including clean data separation, feature engineering, model benchmarking,
final model selection, and reproducible inference.

The project focuses on time series regression, where historical energy
consumption patterns are used to predict future demand.

---

## Problem Statement
Accurate energy demand forecasting is a critical component in modern
energy systems, affecting:
- Power generation planning
- Grid stability
- Operational efficiency
- Cost optimization

Traditional statistical methods often struggle with complex nonlinear
patterns. Therefore, machine learning-based ensemble models are explored.

---

## Objectives
- Build a structured and modular ML pipeline
- Perform time-aware feature engineering
- Train and compare gradient boosting models
- Select a single production-ready model
- Save predictions and evaluation results
- Ensure reproducibility and clarity

---

## Dataset Description
The dataset consists of historical energy consumption data with a
timestamp-based structure.

Data is split into:
- Training set
- Validation set
- Final test set (unseen during training)

Target variable:
- `value` → energy demand

---

## Modeling Approach
Two gradient boosting models are evaluated:
- LightGBM (benchmark)
- XGBoost (final model)

Evaluation metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

---

## Final Results
XGBoost outperformed LightGBM on the test dataset:

| Metric | LightGBM | XGBoost |
|------|---------|--------|
| RMSE | ~18.38 | ~16.95 |
| MAE  | ~4.34  | ~3.33  |
| R²   | ~0.983 | ~0.986 |

XGBoost was selected as the final production model.

---

## Project Structure
- `notebooks/` → step-by-step development
- `src/` → reusable production code
- `models/` → trained models
- `outputs/` → predictions & figures
- `docs/` → documentation

---

## Author
**Gio Maureksa Nugraha**

