import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from model_metrics import *

def train_regression_model(df,test_size=0.2,random_state=42):
    try:
        ##Linear Regression Modeli
        X=df[["Postal Code"]]
        y=df["Sales"]

    
        num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        scaler = StandardScaler()

        # Pipeline: scaler + linear regression
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("linear", LinearRegression())
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        eps = 1e-10  
        smape_value = smape(y_test, y_pred) 
        safe_mape_value=safe_mape(y_test, y_pred, eps=eps)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred) 

        metrics = {"mse": mse, "rmse": rmse, "r2": r2, "smape":"%"+str(smape_value), "safe_mape":"%"+str(safe_mape_value)}
        print(metrics)

        df_encoded = pd.get_dummies(df, columns=["Category"], drop_first=True)

        X = df_encoded[["Category_Office Supplies", "Category_Technology"]]
        y = df_encoded["Sales"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline_encoded = Pipeline([
            ("scaler_encoded", StandardScaler()),
            ("linear_encoded", LinearRegression())
        ])

        pipeline_encoded.fit(X_train, y_train)
        y_pred_encoded = pipeline_encoded.predict(X_test)

        eps = 1e-10  # Small value to avoid division by zero in SMAPE calculation
        smape_value_encoded = smape(y_test, y_pred_encoded) 
        safe_mape_encoded=safe_mape(y_test, y_pred_encoded, eps=eps)
        mse_encoded = mean_squared_error(y_test, y_pred_encoded)
        rmse_encoded = np.sqrt(mse_encoded)
        r2_encoded = r2_score(y_test, y_pred_encoded) 

        metrics_encoded = {"mse": mse_encoded, "rmse": rmse_encoded, "r2": r2_encoded, "smape":"%"+str(smape_value_encoded), "safe_mape":"%"+str(safe_mape_encoded)}
        print(f"Encoding işlemi sonrası metrikler: {metrics_encoded}")

        pipeline3= Pipeline([
            ("scaler3", StandardScaler()),
            ("linear_grid", LinearRegression())
        ])

        param_grid = {
            "linear_grid__fit_intercept": [True, False],
            "linear_grid__positive": [True, False]
        }
        
        grid = GridSearchCV(estimator = pipeline3,
                                param_grid = param_grid,
                                scoring = 'r2',
                                cv = 10,
                                n_jobs = -1)
        
        grid.fit(X_train, y_train)
        best_params = grid.best_params_
        print(f"En iyi parametreler: {best_params}")

        y_pred_grid = grid.predict(X_test)

        smape_value_grid = smape(y_test, y_pred_grid)
        safe_mape_value_grid = safe_mape(y_test, y_pred_grid, eps=eps)
        mse_grid = mean_squared_error(y_test, y_pred_grid)
        rmse_grid = np.sqrt(mse_grid)
        r2_grid = r2_score(y_test, y_pred_grid) 
        metrics_grid = {"mse": mse_grid, "rmse": rmse_grid, "r2": r2_grid, "smape":"%"+str(smape_value_grid), "safe_mape":"%"+str(safe_mape_value_grid)}
        print(f"Grid Search sonrası metrikler: {metrics_grid}")

    except Exception as e:
        print(f"[ERROR] Regresyon modeli eğitilirken hata oluştu: {e}")
        