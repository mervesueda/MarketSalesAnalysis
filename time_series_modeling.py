import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pmdarima import auto_arima
import numpy as np 
import warnings
warnings.filterwarnings("ignore")


#indexed sarima için true, prophet için false - sarıma için çalışmadı
def prepare_dataframe(df,date_col="Order Date", target_col="Sales", freq="D", indexed=False):
    df[date_col] = pd.to_datetime(df[date_col],dayfirst=True)
    df.set_index(date_col, inplace=True)
    df = df.resample(freq).sum().fillna(method="ffill")
    df.reset_index(inplace=not indexed)  # SARIMA için index'li, Prophet için index'siz
    df.rename(columns={date_col: "ds", target_col: "y"}, inplace=True)
    return df

#datetime türündeki sütundan yıl, ay, hafta, gün gibi değişkenleri türetir
def extract_datetime_parts(df, datetime_col):
    try:
        df["Year"] = df[datetime_col].dt.year
        df["Month"] = df[datetime_col].dt.month
        df["Week"] = df[datetime_col].dt.isocalendar().week
        df["Day"] = df[datetime_col].dt.day
        df["DayOfWeek"] = df[datetime_col].dt.dayofweek
        print(f"[INFO] {datetime_col} sütunundan zaman bileşenleri çıkarıldı.")
    except Exception as e:
        print(f"[ERROR] Zaman bileşenleri çıkarılamadı: {e}")
    return df


def adf_test(series):
    try:   
        result = adfuller(series.dropna())
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"   {key}: {value}")

        if result[1] <= 0.05:
            print("[INFO] Zaman serisi durağan. (p-value <= 0.05)")
        else:       
            print("[INFO] Zaman serisi durağan değil. (p-value > 0.05)")

        return result
    
    except Exception as e:
        print(f"[ERROR] ADF testi sırasında hata oluştu: {e}")

def smape(y_true, y_pred,eps=1e-10):
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denom = np.abs(y_true) + np.abs(y_pred)
        denom = np.where(denom == 0, eps, denom)
        smape_value = np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100

        return smape_value
    
    except Exception as e:
        print(f"[ERROR] SMAPE hesaplanırken hata oluştu: {e}")
        return None

def safe_mape(y_true, y_pred, eps=1e-10):
    try:
        y_true=np.asarray(y_true,dtype=float)
        y_pred=np.asarray(y_pred,dtype=float)
        denom=np.where(y_true==0,eps,y_true)
        safe_mape_value = np.mean(np.abs((y_true - y_pred) / denom)) * 100
        return safe_mape_value

    except Exception as e:
        print(f"[ERROR] Safe MAPE hesaplanırken hata oluştu: {e}")
        return None




