import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sales_trend(df, date_col="Order Date", value_col="Sales", freq="D", title="Satış Trendleri", color="#4F81BD"):
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df.set_index(date_col, inplace=True)
        trend = df[value_col].resample(freq).sum()

        plt.figure(figsize=(14, 5))
        plt.plot(trend, color=color)
        plt.title(title)
        plt.xlabel("Tarih")
        plt.ylabel("Toplam Satış")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[ERROR] plot_sales_trend başarısız: {e}")

def plot_correlation(df):
    try:
        # Sadece sayısal sütunları al
        numeric_df = df.select_dtypes(include=["number"])
        
        if numeric_df.empty:
            print("[UYARI] Korelasyon hesaplanacak sayısal sütun yok.")
            return

        corr = numeric_df.corr()  # Korelasyon matrisi
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Korelasyon Matrisi")
        plt.show()
    except Exception as e:
        print(f"[ERROR] plot_correlation başarısız: {e}")


def plot_pie_chart(df, value_col, target_col="Sales", title="Pie Chart", top_n=10):
    grouped = df.groupby(value_col)[target_col].sum().reset_index()
    grouped = grouped.sort_values(by=target_col, ascending=False)

    if len(grouped) > top_n:
        top = grouped.head(top_n)
        other = pd.DataFrame({value_col: ["Diğer"], target_col: [grouped[target_col].iloc[top_n:].sum()]})
        grouped = pd.concat([top, other], ignore_index=True)

    labels = grouped[value_col]
    values = grouped[target_col]

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.axis("equal")  
    plt.tight_layout()
    plt.show()


def plot_subcategory_trend(df):
    try:
        df["Month"] = pd.to_datetime(df["Order Date"], dayfirst=True).dt.to_period("M").astype(str)
        grouped = df.groupby(["Month", "Sub-Category"])["Sales"].sum().reset_index()

        plt.figure(figsize=(18, 6))
        sns.lineplot(data=grouped, x="Month", y="Sales", hue="Sub-Category", marker="o")
        plt.xticks(rotation=45)
        plt.title("Alt Kategori Bazında Aylık Satış Trendleri")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[ERROR] plot_subcategory_trend başarısız: {e}")

def plot_top_cities(df, top_n=10):
    try:
        top_cities = df.groupby("City")["Sales"].sum().sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(16, 6))
        sns.barplot(x=top_cities.index, y=top_cities.values, palette="viridis")
        plt.title(f"En Çok Satış Yapan {top_n} Şehir")
        plt.xlabel("Şehir")
        plt.ylabel("Toplam Satış")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[ERROR] plot_top_cities başarısız: {e}")

def plot_top_products(df, top_n=10):
    try:
        top_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(16, 6))
        sns.barplot(x=top_products.values, y=top_products.index, palette="rocket")
        plt.title(f"En Çok Satılan {top_n} Ürün")
        plt.xlabel("Toplam Satış")
        plt.ylabel("Ürün Adı")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[ERROR] plot_top_products başarısız: {e}")

def plot_sales_kde(df):
    try:
        plt.figure(figsize=(14, 5))
        sns.kdeplot(df["Sales"], fill=True, linewidth=2)
        plt.title("Satış KDE Grafiği")
        plt.xlabel("Satış")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[ERROR] plot_sales_kde başarısız: {e}")

def plot_categorical_kde(df, target="Sales"):
    try:
        for col in df.select_dtypes(include="object").columns:
            if df[col].nunique() < 10:
                plt.figure(figsize=(12, 4))
                sns.kdeplot(data=df, x=target, hue=col, fill=True)
                plt.title(f"{col} için Satış KDE Grafiği")
                plt.tight_layout()
                plt.show()
    except Exception as e:
        print(f"[ERROR] plot_categorical_kde başarısız: {e}")

def plot_lag_and_ma(df, date_col="Order Date", value_col="Sales"):
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
        df.sort_values(date_col, inplace=True)
        df.set_index(date_col, inplace=True)

        # Gecikmeler
        df["lag_1"] = df[value_col].shift(1)
        df["lag_7"] = df[value_col].shift(7)
        df["lag_30"] = df[value_col].shift(30)

        # Hareketli ortalama
        df["MA_30"] = df[value_col].rolling(window=30).mean()

        plt.figure(figsize=(16, 6))
        plt.plot(df[value_col], label=value_col, alpha=0.6)
        plt.plot(df["lag_1"], label="Lag 1", alpha=0.6)
        plt.plot(df["MA_30"], label="MA 30", linewidth=2)
        plt.title("Satış, Gecikmeli ve Hareketli Ortalama Trendleri")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[ERROR] plot_lag_and_ma başarısız: {e}")


def plot_forecast_vs_actual(actual, predicted, lower=None, upper=None, title="Tahmin vs Gerçek"):
    try:
        plt.figure(figsize=(14, 6))
        plt.plot(actual, label="Gerçek Değerler", linewidth=2)
        plt.plot(predicted, label="Tahmin", linestyle="--", linewidth=2)

        if lower is not None and upper is not None:
            plt.fill_between(predicted.index, lower, upper, alpha=0.3, label="Güven Aralığı")

        plt.title(title)
        plt.xlabel("Tarih")
        plt.ylabel("Satış")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[ERROR] plot_forecast_vs_actual başarısız: {e}")

def plot_categorical_violin(df,value_col,target_col="Sales"):
    plt.figure(figsize=(20,8))
    sns.violinplot(x=value_col,y=target_col,data=df,color="#805886")
    plt.title(f"{value_col} - Sales")
    plt.xlabel(value_col)
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




