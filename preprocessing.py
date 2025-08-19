import pandas as pd

#belirtilen sütunları datetime tipine çevirir
def convert_to_datetime(df, columns, dayfirst=True, fmt=None):
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        df[col] = pd.to_datetime(
            df[col].astype(str).str.strip(),
            errors="coerce",
            dayfirst=dayfirst,
            format=fmt  
        )
    return df

#belirtilen sütunları kategorik veriye çevirir
def convert_to_category(df, columns):
    for col in columns:
        try:
            df[col] = df[col].astype("category")
            print(f"[INFO] {col} sütunu kategorik tipe çevrildi.")
        except Exception as e:
            print(f"[ERROR] {col} çevrilemedi: {e}")
    return df

#sütunlardaki eksik değer sayısını verir
def check_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("[INFO] Eksik değer bulunmuyor.")
    else:
        print("[WARNING] Eksik değerler bulundu:")
        print(missing)

#eksik veri içeren satırları siler
def drop_missing_rows(df, how="any", subset=None, inplace=False):
    before = df.shape[0]
    df_cleaned = df.dropna(how=how, subset=subset, inplace=inplace)
    after = df.shape[0] if inplace else df_cleaned.shape[0]
    print(f"[INFO] {before - after} satır eksik veriler nedeniyle silindi.")
    return df if inplace else df_cleaned

#tekrar eden satırları görüntüler 
def show_duplicates(df, subset=None):
    try:
        if df is None or df.empty:
            print("[INFO] DataFrame boş, tekrar eden satır aranmadı.")
            return pd.DataFrame()

        # Tekrar eden satırları bul
        duplicates = df[df.duplicated(subset=subset, keep=False)]

        if duplicates.empty:
            print("[INFO] Tekrar eden satır bulunamadı.")
        else:
            # Kaç benzersiz kaydın tekrar ettiğini say
            duplicate_groups = duplicates.duplicated(subset=subset, keep='first').sum()
            print(f"[WARNING] {duplicates.shape[0]} tekrar eden satır bulundu.")
            print(f"[WARNING] {duplicate_groups} benzersiz kayıt tekrarlanmış.")
            print("[INFO] Tekrar eden satırlar aşağıdadır:")
            print(duplicates.sort_values(by=subset if subset else df.columns.tolist()).to_string(index=False))

        return duplicates

    except KeyError as e:
        print(f"[ERROR] Geçersiz subset sütunu: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Tekrar eden satırlar aranırken hata oluştu: {e}")
        return pd.DataFrame()
    
#tekrar eden satırları siler 
def drop_duplicates_rows(df, subset=None, inplace=False):
    try:
        if df is None or df.empty:
            print("[INFO] DataFrame boş, tekrar eden satır silinmedi.")
            return pd.DataFrame() if not inplace else df

        before = df.shape[0]
        df_cleaned = df.drop_duplicates(subset=subset, inplace=inplace)
        after = df.shape[0] if inplace else df_cleaned.shape[0]
        print(f"[INFO] {before - after} tekrar eden satır silindi.")

        return df if inplace else df_cleaned

    except KeyError as e:
        print(f"[ERROR] Geçersiz subset sütunu: {e}")
        return df
    except Exception as e:
        print(f"[ERROR] Tekrar eden satırlar silinirken hata oluştu: {e}")
        return df

def preprocess_data(df):
    steps = []

    try:
        # 1. Sütun isimlerini düzenle
        old_cols = df.columns.tolist()
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        if old_cols != df.columns.tolist():
            steps.append(f"Sütun isimleri düzenlendi: {old_cols} -> {df.columns.tolist()}")

        # 2. Boş değerleri doldurma
        null_before = df.isnull().sum().sum()
        df = df.fillna("")
        null_after = df.isnull().sum().sum()
        if null_before > null_after:
            steps.append(f"Boş değerler dolduruldu ({null_before} → {null_after})")

        # 3. Gereksiz sütun silme
        cols_to_drop = [col for col in df.columns if "unnamed" in col or col.strip() == ""]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            steps.append(f"Silinen sütunlar: {cols_to_drop}")

        # 4. Adres temizleme
        if "address" in df.columns:
            df["clean_address"] = (
                df["address"]
                .astype(str)
                .str.casefold()
                .str.replace(r"[^\w\s]", " ", regex=True)
                .apply(lambda x: " ".join(x.split()))
            )
            steps.append("`address` sütunu temizlenerek `clean_address` oluşturuldu")

    except Exception as e:
        steps.append(f"Hata: {e}")

    return df, steps
   

def extract_ma_and_lag(df, value_col="Sales", window=30):
    try:
        df["MA_30"] = df[value_col].rolling(window=window).mean()
        df["Lag_1"] = df[value_col].shift(1)
        df["Lag_7"] = df[value_col].shift(7)
        df["Lag_30"] = df[value_col].shift(30)
        print(f"[INFO] {value_col} için hareketli ortalama ve gecikmeli özellikler eklendi.")

        return df
    
    except Exception as e:
        print(f"[ERROR] Hareketli ortalama ve gecikmeli özellikler eklenirken hata oluştu: {e}")
    return df



