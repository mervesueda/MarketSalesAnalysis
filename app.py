import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


# --- Sayfa ayarları ---
st.set_page_config(page_title="Market Sales Analysis", layout="wide", page_icon="📊")

st.markdown("""
Bu uygulama, satış verilerini analiz etmek, ***linear regression modeli*** ve **zaman serisi tahminleri** (SARIMA & Prophet) gerçekleştirmek için geliştirilmiştir.
""")

df = pd.read_csv("train.csv")   # buradaki ismi senin dosyaya göre değiştir
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

@st.cache_data
def load_data():
    df = pd.read_csv("sales.csv")  # kendi dosya adını buraya yaz
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    return df

@st.cache_data
def load_data():
    df = pd.read_csv("sales.csv")  # kendi dosya adını buraya yaz
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    return df

@st.cache_data
def load_data():
    df = pd.read_csv("sales.csv")  # kendi dosya adını buraya yaz
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    return df

df=load_data()


# --- Sidebar Menü ---
menu = st.sidebar.radio(
    "Menü Seçin",
    ["📂 Veri Önizleme", "📊 Keşifsel Analiz", "📈 Tahminler", "⚖️ Model Karşılaştırma"]
)

st.sidebar.markdown("---")
st.sidebar.info("Market Sales Analysis App")

# --- 1. Veri Önizleme ---
if menu == "📂 Veri Önizleme":
    st.title("📂 Veri Önizleme")
    st.dataframe(df.head(20))

    st.write("### Veri Seti Bilgileri")
    st.write(df.describe())

# --- 2. Keşifsel Analiz ---
elif menu == "📊 Keşifsel Analiz":
    st.title("📊 Keşifsel Veri Analizi (EDA)")

    # KPI Kartları
    total_sales = df["Sales"].sum()
    avg_sales = df["Sales"].mean()
    max_cat = df.groupby("Category")["Sales"].sum().idxmax()

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Toplam Satış", f"{total_sales:,.0f}")
    col2.metric("📊 Ortalama Satış", f"{avg_sales:,.2f}")
    col3.metric("🏆 En Çok Satılan Kategori", max_cat)

    # Zaman Serisi Grafiği
    daily_sales = df.groupby("Order Date")["Sales"].sum().reset_index()
    fig = px.line(daily_sales, x="Order Date", y="Sales", title="Zaman İçinde Günlük Satışlar")
    st.plotly_chart(fig, use_container_width=True)

    # Kategori Bazlı Satış
    cat_sales = df.groupby("Category")["Sales"].sum().reset_index()
    fig2 = px.bar(cat_sales, x="Category", y="Sales", title="Kategori Bazlı Satış")
    st.plotly_chart(fig2, use_container_width=True)

# --- 3. Tahminler ---
elif menu == "📈 Tahminler":
    st.title("📈 Zaman Serisi Tahminleri")

    # Tahmin gün sayısı
    periods = st.sidebar.slider("Tahmin Gün Sayısı", 7, 90, 14)

    daily_sales = df.groupby("Order Date")["Sales"].sum().reset_index()
    daily_sales.columns = ["ds", "y"]

    col1, col2 = st.columns(2)

    # Prophet Modeli
    with col1:
        if st.checkbox("Prophet ile Tahmin"):
            prophet = Prophet(daily_seasonality=True)
            prophet.fit(daily_sales)
            future = prophet.make_future_dataframe(periods=periods, freq="D")
            forecast = prophet.predict(future)

            st.write("### Prophet Tahmin Sonuçları", forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(periods))
            fig1 = px.line(forecast, x="ds", y="yhat", title="Prophet Tahmini")
            st.plotly_chart(fig1, use_container_width=True)

    # SARIMA Modeli
    with col2:
        if st.checkbox("SARIMA ile Tahmin"):
            daily_sales2 = daily_sales.copy()
            daily_sales2.set_index("ds", inplace=True)

            model = SARIMAX(daily_sales2["y"], order=(1,1,1), seasonal_order=(1,1,1,7))
            results = model.fit(disp=False)
            forecast_sarima = results.get_forecast(steps=periods)

            st.write("### SARIMA Tahmin Sonuçları", forecast_sarima.predicted_mean)

            fig2, ax = plt.subplots(figsize=(10,5))
            daily_sales2["y"].plot(ax=ax, label="Gerçek")
            forecast_sarima.predicted_mean.plot(ax=ax, label="Tahmin")
            ax.legend()
            st.pyplot(fig2)

# --- 4. Model Karşılaştırma ---
elif menu == "⚖️ Model Karşılaştırma":
    st.title("⚖️ Prophet vs SARIMA Model Karşılaştırması")

    # Eğitim / Test Ayrımı
    daily_sales = df.groupby("Order Date")["Sales"].sum().reset_index()
    daily_sales.columns = ["ds","y"]
    train_size = int(len(daily_sales)*0.8)
    train, test = daily_sales.iloc[:train_size], daily_sales.iloc[train_size:]

    # Prophet
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(train)
    future = prophet.make_future_dataframe(periods=len(test), freq="D")
    forecast_prophet = prophet.predict(future)
    rmse_prophet = np.sqrt(mean_squared_error(test["y"], forecast_prophet.iloc[-len(test):]["yhat"]))

    # SARIMA
    train2 = train.set_index("ds")
    model = SARIMAX(train2["y"], order=(1,1,1), seasonal_order=(1,1,1,7))
    results = model.fit(disp=False)
    forecast_sarima = results.get_forecast(steps=len(test))
    rmse_sarima = np.sqrt(mean_squared_error(test["y"], forecast_sarima.predicted_mean))

    # Tablo Gösterimi
    st.write("### RMSE Karşılaştırması")
    st.table({
        "Model": ["Prophet", "SARIMA"],
        "RMSE": [rmse_prophet, rmse_sarima]
    })

