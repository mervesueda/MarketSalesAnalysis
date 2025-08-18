import pandas as pd
import os

#dosya yolunu kontrol eder
def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    if not path.endswith(".csv"):
        raise ValueError("Yalnızca .csv uzantılı dosyalar desteklenir.")

#eğer dosya yolu çalışıyorsa dosyayı yükler 
def load_data(path):
    check_file_exists(path)
    try:
        df = pd.read_csv(path)
        print(f"[INFO] '{path}' dosyası başarıyla yüklendi.")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("CSV dosyası boş.")
    except pd.errors.ParserError:
        raise ValueError("CSV dosyası okunurken hata oluştu.")
    except Exception as e:
        raise RuntimeError(f"Beklenmeyen bir hata oluştu: {e}")

#dosyadaki veriler hakkında bilgi verir
def inspect_data(df):
    print("\n[INFO] Veri Özeti:")
    print(df.info())
    print("\n[INFO] İlk Satırlar:")
    print(df.head())
    print("\n[INFO] Son Satırlar:")
    print(df.tail())
    print("\n[INFO] Benzersiz Değer Sayısı:")
    print(df.nunique())
    print("\n[INFO] Eksik Değer Sayıları:")
    print(df.isnull().sum())
    print("\n[INFO] Satır-Sütun Sayısı:")
    print(df.shape)
    print("\n[INFO] Sayısal Özellikleri:")
    print(df.describe().T)
    print("\n[INFO] Veriye Ait Sütunlar:")
    print(df.columns)