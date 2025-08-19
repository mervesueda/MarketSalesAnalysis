import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from data_loader import *
from data_visualization import *
from model_metrics import *
from preprocessing import *
from regression_model import *
from time_series_modeling import *
import time


# Sayfa ayarları
st.set_page_config(page_title="Market Sales Analysis", layout="wide", page_icon="📊")
st.title("📊 Market Sales Analysis ")

st.markdown("""
Bu uygulama, satış verilerini analiz etmek, ***linear regression modeli*** ve **zaman serisi tahminleri** (SARIMA & Prophet) gerçekleştirmek için geliştirilmiştir.
""")

df = pd.read_csv("train.csv")   # buradaki ismi senin dosyaya göre değiştir
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# Veri yükleme
@st.cache_data
def get_data():
    path = "train.csv"  # senin dataset
    df = load_data(path)
    return df

df = get_data()

# Sidebar Menü
menu = st.sidebar.radio(
    "Menü Seçin",
    ["📂 Veri Önizleme", "🔧 Ön İşleme", "📊 Görselleştirmeler", "📈 Zaman Serisi Tahminleri", "📉 Regresyon Modeli", "⚖️ Model Karşılaştırma"]
)

st.sidebar.markdown("---")
st.sidebar.info("Market Sales Analysis App")


#1.Veri Temizleme
if menu == "📂 Veri Önizleme":
    st.header("📂 Veri Önizleme")
    st.write("### İlk 10 Satır")
    st.dataframe(df.head(20))
    st.write("### Son 20 Satır")
    st.dataframe(df.tail(10))
    st.write("### Veri Özeti")
    st.write(df.describe(include="all"))


# 2.Ön işleme
elif menu == "🔧 Ön İşleme":
    if st.button("🚀 Veri Ön İşlemeyi Başlat"):
        with st.spinner("Veri ön işleme başlatılıyor..."):
            progress_text = "Veri ön işleniyor..."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(0, 101, 20):
                time.sleep(0.5)
                my_bar.progress(percent_complete, text=progress_text)

            try:
                df_clean, steps = preprocess_data(df)   # ✅ iki değer yakala
                st.success("✅ Veri ön işleme tamamlandı!")
                st.subheader("İşlenmiş Veri Önizleme")
                st.dataframe(df_clean.head())

                # Yapılan işlemleri göster
                st.subheader("🔎 Yapılan İşlemler")
                for step in steps:
                    st.write("•", step)

                # İndirme seçeneği
                csv = df_clean.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 İşlenmiş Veriyi İndir",
                    data=csv,
                    file_name="clean_data.csv",
                    mime="text/csv"
                )

            except Exception:
                st.error("❌ Veri ön işleme sırasında bir hata oluştu.")



# 3. görselleştirme
elif menu == "📊 Görselleştirmeler":
    st.header("📊 Keşifsel Veri Görselleştirme")

    st.subheader("Korelasyon Isı Haritası")
    plot_correlation(df)

    st.subheader("Kategorilere Göre Satış Dağılımı")
    plot_pie_chart(df, label_col="Category", value_col="Sales", title="Kategori - Satış")

# 4.Zaman serisi
elif menu == "📈 Zaman Serisi Tahminleri":
    st.header("📈 Zaman Serisi Tahminleri")

    # Prophet Zaman Serisi Modeli
    st.subheader("📌 Prophet Modeli")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df_prophet = df.groupby("Order Date")["Sales"].sum().reset_index()
    df_prophet.columns = ["ds", "y"]

    train_size_prophet = int(len(df_prophet) * 0.7)
    train_prophet = df_prophet[:train_size_prophet]
    test_prophet = df_prophet[train_size_prophet:]

    prophet = Prophet()
    prophet.fit(train_prophet)

    future_prophet = prophet.make_future_dataframe(periods=7, freq="D")
    forecast_prophet = prophet.predict(future_prophet)

    st.write("Prophet Tahmin Tablosu (Son 7 Gün)")
    st.dataframe(forecast_prophet[["ds","yhat","yhat_lower","yhat_upper"]].tail(7))

    fig1 = prophet.plot(forecast_prophet)
    plt.title("Prophet Tahmin Sonuçları")
    plt.xlabel("Tarih")
    plt.ylabel("Satış")
    st.pyplot(fig1)

    fig2 = prophet.plot_components(forecast_prophet)
    st.pyplot(fig2)

    # SARIMA Modeli
    st.subheader("📌 SARIMA Modeli")
    df_sarima = df.groupby("Order Date")["Sales"].sum().reset_index()
    df_sarima.set_index("Order Date", inplace=True)

    full_range = pd.date_range(start=df_sarima.index.min(), end=df_sarima.index.max(), freq="D")
    df_sarima = df_sarima.reindex(full_range)
    df_sarima.index.name = "Order Date"
    df_sarima = df_sarima.fillna(0)

    train_size_sarima = int(len(df_sarima) * 0.7)
    train_sarima = df_sarima[:train_size_sarima]
    test_sarima = df_sarima[train_size_sarima:]

    from pmdarima import auto_arima
    stepwise_model = auto_arima(train_sarima["Sales"],
                                start_p=1, start_q=1,
                                max_p=3, max_q=3,
                                d=None,
                                start_P=0, seasonal=True,
                                D=1, m=7,
                                trace=False,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

    p,d,q = stepwise_model.order
    P,D,Q,m = stepwise_model.seasonal_order

    sarima = SARIMAX(endog=train_sarima["Sales"],
                     order=(p, d, q),
                     seasonal_order=(P, D, Q, m),
                     enforce_invertibility=False,
                     enforce_stationarity=False)

    results_sarima = sarima.fit()
    forecast_sarima = results_sarima.get_forecast(7)
    forecast_sarima_mean = forecast_sarima.predicted_mean.to_frame(name="yhat_sarima")

    ci = forecast_sarima.conf_int().copy()
    ci.columns = ["yhat_lower","yhat_upper"]
    forecast_sarima_mean.index = test_sarima.index[:7]
    ci.index = test_sarima.index[:7]

    # Çizim
    fig3, ax = plt.subplots(figsize=(12,6))
    df_sarima["Sales"].plot(ax=ax, label="Gerçek Satış", color="#61AC80")
    forecast_sarima.predicted_mean.plot(ax=ax, label="SARIMA Tahmin", color="#487D95")
    ax.fill_between(ci.index,
                    ci.iloc[:, 0],
                    ci.iloc[:, 1],
                    color="#487D95", alpha=0.2)
    plt.title("SARIMA Forecast vs Sales")
    plt.legend()
    st.pyplot(fig3)


 # Zaman serisi metrikleri
    st.subheader("📊 Model Performans Metrikleri (7 Günlük)")

    # Prophet metrikleri
    steps = 7
    y_true7 = test_prophet.set_index("ds")["y"].iloc[:steps]
    y_pred7 = forecast_prophet.set_index("ds")["yhat"].loc[
        y_true7.index.intersection(forecast_prophet["ds"])
    ]
    y_pred_prophet = y_pred7.reindex(y_true7.index).dropna()
    y_true_prophet = y_true7.loc[y_pred7.index]

    # SARIMA metrikleri
    y_true_sarima = test_sarima["Sales"].iloc[:7]
    y_pred_sarima = forecast_sarima.predicted_mean
    y_pred_sarima.index = y_true_sarima.index

    rmse_prophet = np.sqrt(mean_squared_error(y_true_prophet, y_pred_prophet))
    rmse_sarima = np.sqrt(mean_squared_error(y_true_sarima, y_pred_sarima))

    smape_prophet = smape(y_true_prophet, y_pred_prophet)
    smape_sarima = smape(y_true_sarima, y_pred_sarima)

    r2_prophet = r2_score(y_true_prophet, y_pred_prophet)
    r2_sarima = r2_score(y_true_sarima, y_pred_sarima)

    # Streamlit tablosu
    metrics_df = pd.DataFrame({
        "Model": ["Prophet", "SARIMA"],
        "RMSE": [rmse_prophet, rmse_sarima],
        "SMAPE": [smape_prophet, smape_sarima],
        "R2": [r2_prophet, r2_sarima]
    })
    st.dataframe(metrics_df)

# 5. Regresyon modeli
elif menu == "📉 Regresyon Modeli":
    st.header("📉 Regresyon Modeli Performansı")

    try:
        result = train_regression_model(df)

        # Eğer fonksiyon dict döndürüyorsa doğrudan al
        if isinstance(result, dict):
            metrics = result
        else:
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression

            X = df[["Postal Code"]]
            y = df["Sales"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": root_mean_squared_error(y_test, y_pred),  # model_metrics.py'den
                "R2": r2_score(y_test, y_pred),
                "SMAPE": smape(y_test, y_pred)                   # model_metrics.py'den
            }

        st.subheader("🔎 Regresyon Modeli Metrikleri")
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df)

    except Exception as e:
        st.error(f"Regresyon modeli çalıştırılırken hata oluştu: {e}")


 
        



