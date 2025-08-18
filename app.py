import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score

#Sayfa Başlığı ve Açıklama
st.set_page_config(page_title="Market Sales Analysis", layout="wide")

st.title("📊 Market Sales Analysis Dashboard")
st.markdown("""
Bu uygulama, satış verilerini analiz etmek, linear regression modeli ve **zaman serisi tahminleri** (SARIMA & Prophet) gerçekleştirmek için geliştirilmiştir.
""")

df = pd.read_csv("train.csv")   # buradaki ismi senin dosyaya göre değiştir
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

st.write("### Veri Önizleme")
st.dataframe(df.head())
