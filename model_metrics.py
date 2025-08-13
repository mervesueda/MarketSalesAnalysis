import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score,confusion_matrix
from time_series_modeling import smape
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def safe_mape(y_true, y_pred, eps=1e-10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    denom=np.where(y_true == 0, eps, y_true) #koşullu değer atama var, eğer y ture 0 ise eps değilde y_true kullanılıyor

    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def smape(y_true, y_pred, eps=1e-10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("[ERROR] Şekiller aynı olmalı")
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, eps, denom)
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100

