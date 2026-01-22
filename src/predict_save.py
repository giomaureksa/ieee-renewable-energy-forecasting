import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

def predict_and_save(model_dir, test_path, results_dir):
    print("Loading test data...")
    test = pd.read_csv(test_path, parse_dates=["timestamp"])
    
    target = "value"
    features = [col for col in test.columns if col not in ["timestamp", "value", "series_id"]]
    
    X_test = test[features]
    y_test = test[target] if target in test.columns else None
    
    print("Loading models...")
    lgb_model = lgb.Booster(model_file=f"{model_dir}/lgb_model.txt")
    xgb_model = xgb.Booster()
    xgb_model.load_model(f"{model_dir}/xgb_model.json")
    
    print("Making predictions...")
    lgb_pred = lgb_model.predict(X_test)
    dtest = xgb.DMatrix(X_test)
    xgb_pred = xgb_model.predict(dtest)
    
    if y_test is not None:
        lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
        lgb_mae = mean_absolute_error(y_test, lgb_pred)
        lgb_r2 = r2_score(y_test, lgb_pred)
        
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        
        print("LightGBM Test Performance:")
        print(f"RMSE: {lgb_rmse:.4f}, MAE: {lgb_mae:.4f}, R²: {lgb_r2:.4f}")
        print("XGBoost Test Performance:")
        print(f"RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}, R²: {xgb_r2:.4f}")
    
    print("Saving predictions...")
    pred_df = pd.DataFrame({
        "series_id": test["series_id"],
        "timestamp": test["timestamp"],
        "value_lgb": lgb_pred,
        "value_xgb": xgb_pred
    })
    
    os.makedirs(results_dir, exist_ok=True)
    pred_df.to_csv(f"{results_dir}/test_predictions.csv", index=False)
    
    if y_test is not None:
        plt.figure(figsize=(12,6))
        plt.plot(test["timestamp"], y_test, label="Actual", alpha=0.7)
        plt.plot(test["timestamp"], lgb_pred, label="LightGBM", alpha=0.7)
        plt.plot(test["timestamp"], xgb_pred, label="XGBoost", alpha=0.7)
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.title("Test Set Predictions vs Actual")
        plt.legend()
        plt.savefig(f"{results_dir}/figures/prediction_plot.png", dpi=300)
        plt.show()
    
    print("Predictions saved.")

if __name__ == "__main__":
    predict_and_save(
        "../models",
        "../data/processed/test_features.csv",
        "../results"
    )