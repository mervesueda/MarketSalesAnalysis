import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error,root_mean_squared_error
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
from prophet import Prophet
    


from preprocessing import *
from data_visualization import *
from time_series_modeling import *
from model_metrics import *
from regression_model import train_regression_model

from data_loader import (
    load_data,
    inspect_data
)


import warnings
warnings.filterwarnings("ignore")

#Veri yükleme
path = "train.csv"
df = load_data(path)
inspect_data(df)

#Veri işleme
if df is not None:
    df = convert_to_datetime(df, "Order Date", dayfirst=True, fmt="%d/%m/%Y")
    df = convert_to_datetime(df, "Ship Date", dayfirst=True, fmt="%d/%m/%Y")
    df = drop_missing_rows(df)
    cols_to_remove=["Row ID","Order ID","Customer ID","Product ID","Country"]
    df=df.drop(columns=cols_to_remove, errors="ignore")
    df = convert_to_category(df, ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"])

    #Object olan kategorik sütunların kategoriye dönüşümü,hareketli ort, lag features 
    print(df.info())
    df=extract_ma_and_lag(df)
    df=extract_datetime_parts(df=df,datetime_col="Order Date")

#Temizlik Sonrası Data Kontrol
inspect_data(df)

#Prophet Zaman Serisi Modeli
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
df_prophet = df.groupby("Order Date")["Sales"].sum().reset_index()
df_prophet.columns = ["ds", "y"]

train_size_prophet = int(len(df_prophet) * 0.7)
train_prophet = df_prophet[:train_size_prophet]
test_prophet = df_prophet[train_size_prophet:]

prophet=Prophet()
prophet.fit(train_prophet)  


future_prophet = prophet.make_future_dataframe(
    periods=7, 
    freq='D' 
)

forecast_prophet = prophet.predict(future_prophet)


prophet.plot_components(forecast_prophet)
plt.show()

#Prophet için tahmin sonuçları görselleştirme - visualization a fonk. olarak koyulacak!!!
prophet.plot(forecast_prophet)
plt.title("Prophet Tahmin Sonuçları")
plt.xlabel("Tarih")
plt.ylabel("Satış")
plt.legend()
plt.tight_layout()
plt.show()

prophet.plot_components(forecast_prophet)
plt.show()


#sarima deneme
df_sarima = df.groupby("Order Date")["Sales"].sum().reset_index()
df_sarima.set_index("Order Date", inplace=True)

full_range = pd.date_range(start=df_sarima.index.min(), end=df_sarima.index.max(), freq="D")
df_sarima = df_sarima.reindex(full_range)
df_sarima.index.name = "Order Date"
df_sarima = df_sarima.fillna(0)

train_size_sarima= int(len(df_sarima) * 0.7)
train_sarima = df_sarima[:train_size_sarima]
test_sarima = df_sarima[train_size_sarima:]

adf_test(df_sarima["Sales"])

stepwise_model=auto_arima(train_sarima["Sales"],
                             start_p=1, start_q=1,      
                             max_p=3, max_q=3,          
                             d=None,                    
                             start_P=0, seasonal=True,  
                             D=1,                       
                             m=7,                       
                             trace=True,                
                             error_action='ignore',     
                             suppress_warnings=True,    
                             stepwise=True)  

print(stepwise_model.summary())

#stepwise ile belirlenen değerleri alır 
p,d,q = stepwise_model.order
P,D,Q,m = stepwise_model.seasonal_order

#normalde modeli df_sarima ile eğitmiştim fakat veri sızıntısı olmasın diye train_sarima ile eğitmeyi deniyorum
#ndog parametresi verilen tüm parametreyi eğltim olarak kabul eder
sarima=SARIMAX(endog=train_sarima["Sales"],
              order=(p, d, q),
              seasonal_order=(P, D, Q, m),
              enforce_invertibility=False,
              enforce_stationarity=False
            )

results_sarima = sarima.fit()
forecast_sarima= results_sarima.get_forecast(7)  
forecast_sarima_mean = forecast_sarima.predicted_mean.to_frame(name="yhat_sarima")
forecast_sarima_mean.rename(columns={'index': 'ds', 'mean': 'yhat_sarima'}, inplace=True)

ci = forecast_sarima.conf_int().copy() 
ci.columns=["yhat_lower","yhat_upper"]
forecast_sarima_mean.index = test_sarima.index[:7]
ci.index = test_sarima.index[:7]

# Gerçek değerleri çiz
ax = df_sarima['Sales'].plot(label='Sales', figsize=(20, 6),color="#61AC80")

# SARIMA tahminlerini çiz
forecast_sarima.predicted_mean.plot(ax=ax, label='SARIMA Predict', color="#487D95")

# Güven aralığını doldur
ax.fill_between(ci.index,
                ci.iloc[:, 0],
                ci.iloc[:, 1],
                color='#487D95', alpha=0.2)

forecast_end = ci.index[-1]  # Tahminin son günü
ax.set_xlim(df_sarima.index.min(), forecast_end)

plt.title("SARIMA Forecast vs Sales")
plt.legend()
plt.show()# Gerçek değerleri çiz
ax = df_sarima['Sales'].plot(label='Sales', figsize=(20, 6))

#Regresyon Modeli
train_regression_model(df)


#zaman serisi metrikleri
steps = 7
y_true7 = test_prophet.set_index("ds")["y"].iloc[:steps]
y_pred7 = forecast_prophet.set_index("ds")["yhat"].loc[y_true7.index.intersection(
    forecast_prophet["ds"]
)]
y_pred_prophet = y_pred7.reindex(y_true7.index).dropna()
y_true_prophet = y_true7.loc[y_pred7.index]
rmse_prophet = np.sqrt(mean_squared_error(y_true7, y_pred7))


y_true_sarima = test_sarima["Sales"].iloc[:7]
y_pred_sarima = forecast_sarima.predicted_mean            
y_pred_sarima.index = y_true_sarima.index   # hizala

rmse_prophet = root_mean_squared_error(y_true_prophet, y_pred_prophet)
rmse_sarima = root_mean_squared_error(y_true_sarima, y_pred_sarima)

smape_prophet = smape(y_true_prophet, y_pred_prophet)
smape_sarima = smape(y_true_sarima, y_pred_sarima) 

r2_prophet = r2_score(y_true_prophet, y_pred_prophet)
r2_sarima = r2_score(y_true_sarima, y_pred_sarima)

print(f"Prophet RMSE: {rmse_prophet:.2f}, SMAPE: %{smape_prophet:.2f}, R2:{r2_prophet:.2f}") 
print(f"SARIMA  RMSE: {rmse_sarima:.2f}, SMAPE: %{smape_sarima:.2f}, R2:{r2_sarima:.2f}")

#Günlük - Haftalık _aylık Satış Trendleri
plot_sales_trend(df, freq="D",title="Günlük Satış Trendleri")
plot_sales_trend(df, freq="W",title="Haftalık Satış Trendleri")
plot_sales_trend(df, freq="M",title="Aylık Satış Trendleri")

#ilk 10 şehir - ürün / alt kategorik değişkenler satış grafiği
plot_subcategory_trend(df)
plot_top_products(df)
plot_top_cities(df)

#lag değişkenler için scatter plot
lag_columns=["lag_1","lag_7","lag_30"]
plot_lag_and_ma(df,date_col="Order Date", value_col="Sales")

#korelasyon grafiği
plot_correlation(df)

#Pie chart
pie_cols=["Ship Mode", "Segment", "Region", "Category", "Year"]
for c in pie_cols:
    plot_pie_chart(df, value_col=c, target_col="Sales", title=f"{c} - Sales", top_n=8)

#kategorik değişkenler için violin plot
df_filtered_cols=[col for col in df if df[col].nunique()<=10 and df[col].dtype.name=="category"]
for c in df_filtered_cols:
    plot_categorical_violin(df,c,"Sales")






