import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,root_mean_squared_error
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
    ["📂 Veri Önizleme", "🔧 Ön İşleme", "📊 Görselleştirmeler", "📈 Zaman Serisi Tahminleri", "📉 Regresyon Modeli"]
)

st.sidebar.markdown("---")
st.sidebar.info("Market Sales Analysis App")


#1.Veri Temizleme
if menu == "📂 Veri Önizleme":
    st.header("📂 Veri Önizleme")
    st.write("### İlk 10 Satır")
    st.dataframe(df.head(10))
    st.write("### Son 10 Satır")
    st.dataframe(df.tail(10))
    st.write("### Veri Özeti")
    st.write(df.describe(include="all"))

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



elif menu == "📊 Görselleştirmeler":
    st.header("📊 Keşifsel Veri Görselleştirme")

    # Grafik seçenekleri (kategoriye göre ayrıldı)
    trend_grafikleri = [
        "📈 Günlük Satış Trendleri",
        "📉 Haftalık Satış Trendleri",
        "📊 Aylık Satış Trendleri",
        "🔗 Lag Features ve MA Değişkenini Gösteren Grafik"
    ]

    dagilim_grafikleri = [
        "🔥 Korelasyon Isı Haritası",
        "🥧 Kategorilere Göre Satış Dağılımı",
        "🥧 Segmentlere Göre Satış Dağılımı",
        "🥧 Bölgelere Göre Satış Dağılımı",
        "🥧 Yıllara Göre Satış Dağılımı",
        "🥧 Sezonlara Göre Satış Dağılımı",
        "📊 Alt Kategori Bazında Aylık Satış Dağılımı",
        "🏙️ En Çok Satış Yapan 10 Şehir",
        "📦 En Çok Satılan 10 Ürün",
        "🎻 Violin Plot",
        "📊 KDE Grafiği"
    ]

    kategori = st.radio("Grafik kategorisini seçin:", ["📈 Trend Grafikleri", "📊 Dağılım Grafikleri"])

    if kategori == "📈 Trend Grafikleri":
        secilen_grafik = st.selectbox("Trend grafiğini seçin:", trend_grafikleri)
    else:
        secilen_grafik = st.selectbox("Dağılım grafiğini seçin:", dagilim_grafikleri)

    if st.button("📊 Grafiği Göster"):
        if secilen_grafik == "📈 Günlük Satış Trendleri":
            fig = plot_sales_trend(df, "Order Date", "Sales", "D", "Günlük Satış Trendleri", "#4F81BD")
            st.pyplot(fig)

        elif secilen_grafik == "📉 Haftalık Satış Trendleri":
            fig = plot_sales_trend(df, "Order Date", "Sales", "W", "Haftalık Satış Trendleri", "#6B8F81")
            st.pyplot(fig)

        elif secilen_grafik == "📊 Aylık Satış Trendleri":
            fig = plot_sales_trend(df, "Order Date", "Sales", "M", "Aylık Satış Trendleri", "#7B17CE")
            st.pyplot(fig)

        elif secilen_grafik == "🔥 Korelasyon Isı Haritası":
            fig = plot_correlation(df)
            st.pyplot(fig)

        elif secilen_grafik == "🥧 Kategorilere Göre Satış Dağılımı":
            fig = plot_pie_chart(df, "Category", "Sales", "Kategori - Satış")
            st.pyplot(fig)

        elif secilen_grafik == "🥧 Segmentlere Göre Satış Dağılımı":
            fig = plot_pie_chart(df, "Segment", "Sales", "Segment - Satış")
            st.pyplot(fig)

        elif secilen_grafik == "🥧 Bölgelere Göre Satış Dağılımı":
            fig = plot_pie_chart(df, "Region", "Sales", "Bölge - Satış")
            st.pyplot(fig)

        elif secilen_grafik == "🥧 Yıllara Göre Satış Dağılımı":
            fig = plot_pie_chart(df, "Year", "Sales", "Yıl - Satış")
            st.pyplot(fig)

        elif secilen_grafik == "🥧 Sezonlara Göre Satış Dağılımı":
            fig = plot_pie_chart(df, "Season", "Sales", "Sezon - Satış")
            st.pyplot(fig)

        elif secilen_grafik == "📊 Alt Kategori Bazında Aylık Satış Dağılımı":
            fig = plot_subcategory_trend(df)
            st.pyplot(fig)

        elif secilen_grafik == "🏙️ En Çok Satış Yapan 10 Şehir":
            fig = plot_top_cities(df, top_n=10)
            st.pyplot(fig)

        elif secilen_grafik == "📦 En Çok Satılan 10 Ürün":
            fig = plot_top_products(df, top_n=10)
            st.pyplot(fig)

        elif secilen_grafik == "📊 KDE Grafiği":
            fig = plot_sales_kde(df)
            st.pyplot(fig)

        elif secilen_grafik == "🔗 Lag Features ve MA Değişkenini Gösteren Grafik":
            fig = plot_lag_and_ma(df)
            st.pyplot(fig)

        elif secilen_grafik == "🎻 Violin Plot":
            df_filtered_cols = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtype == "object"]
            for c in df_filtered_cols:
                fig = plot_categorical_violin_for_streamlit(df, c, "Sales")
                st.pyplot(fig)

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
    
    try:
        df_sarima = df.groupby("Order Date")["Sales"].sum().reset_index()
        df_sarima.set_index("Order Date", inplace=True)

        full_range = pd.date_range(start=df_sarima.index.min(), end=df_sarima.index.max(), freq="D")
        df_sarima = df_sarima.reindex(full_range)
        df_sarima.index.name = "Order Date"
        df_sarima = df_sarima.fillna(0)

        train_size_sarima = int(len(df_sarima) * 0.7)
        train_sarima = df_sarima[:train_size_sarima]
        test_sarima = df_sarima[train_size_sarima:]

        # pmdarima kütüphanesini kontrol et
        try:
            from pmdarima import auto_arima
        except ImportError:
            st.error("❌ pmdarima kütüphanesi yüklü değil. Lütfen 'pip install pmdarima' komutu ile yükleyin.")
            st.stop()

        # Spinner ile birlikte auto_arima
        with st.spinner("SARIMA parametreleri optimize ediliyor... ⏳"):
            try:
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
                
                st.success(f"✅ En iyi SARIMA parametreleri bulundu: {stepwise_model.order} x {stepwise_model.seasonal_order}")
                
            except Exception as e:
                st.error(f"❌ Auto ARIMA hatası: {str(e)}")
                # Fallback parametreler
                st.warning("⚠️ Varsayılan parametreler kullanılacak...")
                p, d, q = 1, 1, 1
                P, D, Q, m = 1, 1, 1, 7
            else:
                p, d, q = stepwise_model.order
                P, D, Q, m = stepwise_model.seasonal_order

        # SARIMA modelini fit et
        with st.spinner("SARIMA modeli eğitiliyor... ⏳"):
            try:
                sarima = SARIMAX(endog=train_sarima["Sales"],
                                 order=(p, d, q),
                                 seasonal_order=(P, D, Q, m),
                                 enforce_invertibility=False,
                                 enforce_stationarity=False)

                results_sarima = sarima.fit(disp=False)  # disp=False eklendi
                st.success("✅ SARIMA modeli başarıyla eğitildi!")
                
            except Exception as e:
                st.error(f"❌ SARIMA model eğitim hatası: {str(e)}")
                st.stop()

    except Exception as e:
        st.error(f"❌ SARIMA model eğitim hatası: {str(e)}")
        st.stop()

    # Tahmin yap
    try:
        forecast_sarima = results_sarima.get_forecast(7)
        forecast_sarima_mean = forecast_sarima.predicted_mean.to_frame(name="yhat_sarima")

        # SARIMA Tahmin tablosunu göster
        st.write("SARIMA Tahmin Tablosu (Gelecek 7 Gün)")
        forecast_display = forecast_sarima_mean.copy()
        forecast_display["Tarih"] = forecast_display.index.strftime('%Y-%m-%d')
        forecast_display = forecast_display[["Tarih", "yhat_sarima"]].round(2)
        st.dataframe(forecast_display)

        ci = forecast_sarima.conf_int().copy() 
        ci.columns = ["yhat_lower", "yhat_upper"]

            # Çizim - GERÇEK + TAHMİN (sadece tahmin sonuna kadar)
        fig3, ax = plt.subplots(figsize=(12, 6))

        # Gerçek satışlar (tahmin bitişine kadar)
        end_date = forecast_sarima_mean.index[-1]
        df_sarima.loc[:end_date, "Sales"].plot(ax=ax, label="Gerçek Satış", color="#61AC80", linewidth=2)

        # SARIMA tahmini
        forecast_sarima_mean["yhat_sarima"].plot(ax=ax, label="SARIMA Tahmin", 
                                                color="#487D95", linewidth=2, linestyle="--")

        # Güven aralığı
        ax.fill_between(ci.index, ci["yhat_lower"], ci["yhat_upper"], 
                        color="#487D95", alpha=0.2, label="Güven Aralığı")

        # Görsel ayarları
        ax.set_xlim(df_sarima.index.min(), end_date)   # sadece tahminin bittiği yere kadar göster
        ax.set_title("SARIMA Tahmin Sonuçları", fontsize=14, fontweight="bold")
        ax.set_xlabel("Tarih", fontsize=12)
        ax.set_ylabel("Satış", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        st.pyplot(fig3)


    except Exception as e:
        st.error(f"❌ SARIMA tahmin hatası: {str(e)}")

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
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression

        # ✅ Ön işlem yapılmış veriyi kullan (session_state'ten al)
        if "df_clean" in st.session_state:
            df_reg = st.session_state.df_clean
        else:
            df_reg = df.dropna(subset=["Postal Code"])  # fallback

        # Özellikler ve hedef değişken
        X = df_reg[["Postal Code"]]
        y = df_reg["Sales"]

        # Eğitim / test ayrımı
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model kurulumu ve eğitimi
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Tahmin
        y_pred = model.predict(X_test)

        # Metrikler
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
            "SMAPE": smape(y_test, y_pred)
        }

        # Sonuçların tablo halinde gösterilmesi
        st.subheader("🔎 Regresyon Modeli Metrikleri")
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df)

    except Exception as e:
        st.error(f"Regresyon modeli çalıştırılırken hata oluştu: {e}")