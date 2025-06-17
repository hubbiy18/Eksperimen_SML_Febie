
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import zscore

def preprocess_diabetes_dataset(filepath, save_cleaned=True):
    """
    Fungsi untuk memuat dan melakukan preprocessing dataset diabetes.
    
    Tahapan preprocessing:
    1. Load dataset
    2. Drop duplikat
    3. Encode data kategorikal
    4. Deteksi & hapus outlier (Z-score)
    5. Normalisasi fitur numerik
    6. Simpan data bersih ke file CSV (opsional)
    7. Return dataframe bersih
    """

    df = pd.read_csv(filepath)
    df = df.drop_duplicates()

    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])

    le_smoking = LabelEncoder()
    df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])

    numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi',
                    'HbA1c_level', 'blood_glucose_level', 'diabetes']
    
    z_scores = df[numeric_cols].apply(zscore)
    df = df[(z_scores.abs() <= 3).all(axis=1)]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    if save_cleaned:
        df.to_csv("diabetes_cleaned.csv", index=False)

    return df
