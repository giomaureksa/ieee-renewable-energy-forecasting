import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json

def train_and_evaluate(train_path, val_path, model_dir):
    print("Loading data...")
    train = pd.read_csv(train_path, parse_dates=["timestamp"])
    val = pd.read_csv(val_path, parse_dates=["timestamp"])
    
    target = "value"
    features = [col for col in train.columns if col not in ["timestamp", "value", "series_id"]]
    
    X_train = train[features]
    y_train = train[target]
    X_val = val[features]
    y_val = val[target]
    
    print("Training LightGBM...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1
    }
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    print("Training XGBoost...")
    xgb_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "eval_metric": "rmse"
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    print("Evaluating models...")
    lgb_pred = lgb_model.predict(X_val)
    xgb_pred = xgb_model.predict(dval)
    
    lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    lgb_r2 = r2_score(y_val, lgb_pred)
    
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
    xgb_mae = mean_absolute_error(y_val, xgb_pred)
    xgb_r2 = r2_score(y_val, xgb_pred)
    
    print("LightGBM Performance:")
    print(f"RMSE: {lgb_rmse:.4f}, MAE: {lgb_mae:.4f}, R²: {lgb_r2:.4f}")
    print("XGBoost Performance:")
    print(f"RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}, R²: {xgb_r2:.4f}")
    
    print("Saving models...")
    lgb_model.save_model(f"{model_dir}/lgb_model.txt")
    xgb_model.save_model(f"{model_dir}/xgb_model.json")
    with open(f"{model_dir}/xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    
    print("Training complete.")

if __name__ == "__main__":
    train_and_evaluate(
        "../data/processed/train_features.csv",
        "../data/processed/validation_features.csv",
        "../models"
    )