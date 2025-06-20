# File: preprocessing/automate_febie.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Contoh model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import os

def preprocess_diabetes_dataset(filepath, save_cleaned_path=None):
    """
    Fungsi untuk memuat dan melakukan preprocessing dataset diabetes.
    
    Tahapan preprocessing:
    1. Load dataset
    2. Drop duplikat
    3. Tangani missing values
    4. Encode data kategorikal
    5. Deteksi & hapus outlier (Z-score)
    6. Normalisasi fitur numerik
    7. Simpan data bersih ke file CSV (opsional)
    8. Return dataframe bersih
    """

    # 1. Load dataset
    df = pd.read_csv(filepath)
    print(f"Data awal: {df.shape[0]} baris, {df.shape[1]} kolom")

    # 2. Drop duplicates
    df = df.drop_duplicates()
    print(f"Setelah hapus duplikat: {df.shape[0]} baris")

    # 3. Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    print("Missing values ditangani.")

    # 4. Encode categorical features
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    print("Fitur kategorikal di-encode.")

    # 5. Remove outliers with Z-score (exclude binary/target cols)
    exclude_cols = ['hypertension', 'heart_disease', 'diabetes', 'gender'] 
    numeric_cols = df.select_dtypes(include=np.number).columns.difference(exclude_cols)
    
    if not numeric_cols.empty:
        z_scores = df[numeric_cols].apply(zscore)
        initial_rows = df.shape[0]
        # Filter DataFrame berdasarkan Z-score
        df = df[(np.abs(z_scores) < 3).all(axis=1)]
        print(f"Setelah hapus outlier: {df.shape[0]} baris (dihapus {initial_rows - df.shape[0]} baris)")
    else:
        print("Tidak ada kolom numerik yang cocok untuk deteksi outlier (setelah mengabaikan kolom pengecualian).")


    # 6. Feature scaling
    features_to_scale = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    # Pastikan kolom ada sebelum scaling
    actual_features_to_scale = [f for f in features_to_scale if f in df.columns]
    
    if actual_features_to_scale:
        scaler = StandardScaler()
        df[actual_features_to_scale] = scaler.fit_transform(df[actual_features_to_scale])
        print("Fitur numerik discale.")
    else:
        print("Tidak ada fitur yang cocok untuk scaling.")


    # 7. Save cleaned dataset
    if save_cleaned_path:
        # Pastikan direktori output ada
        output_dir = os.path.dirname(save_cleaned_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_csv(save_cleaned_path, index=False)
        print(f"Data bersih disimpan ke '{save_cleaned_path}'")

    # 8. Return dataframe
    print("Distribusi label diabetes:\n", df['diabetes'].value_counts())
    return df

# --- Bagian Baru: Fungsi untuk Melatih dan Log Model ---
def train_and_log_diabetes_model(df_cleaned):
    """
    Fungsi untuk melatih model klasifikasi diabetes dan mencatatnya ke MLflow.
    Menerima dataframe yang sudah bersih.
    """
    print("\nMemulai proses pelatihan dan logging model...")

    # Pisahkan fitur (X) dan target (y)
    X = df_cleaned.drop('diabetes', axis=1)
    y = df_cleaned['diabetes']

    # Bagi data menjadi training dan testing
    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Data dibagi: {len(X_train)} sampel training, {len(X_test)} sampel testing")

    with mlflow.start_run(run_name="Diabetes_Prediction_RF_Model_Combined"):
        print("MLflow run dimulai...")

        # Definisikan hyperparameter model Random Forest
        n_estimators = 100
        max_depth = 10
        
        # Log Hyperparameter
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        print("Hyperparameter dilog.")

        # Inisialisasi dan latih model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        print("Model dilatih.")

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] # Probability for ROC AUC

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Log Metrik
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        print(f"Metrik dilog: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC_AUC={roc_auc:.4f}")
        
        # Log Model ke MLflow
        mlflow.sklearn.log_model(
            model, 
            "diabetes_rf_model", 
            registered_model_name="DiabetesPredictionClassifierCombined"
        )
        print("Model berhasil dilog ke MLflow!")

if __name__ == "__main__":
    # Path dataset mentah relatif terhadap root repo
    # Karena script ini ada di preprocessing/, maka harus naik 1 level untuk ke root repo
    # Dan diabetes_prediction_dataset.csv ada di root repo
    current_dir = os.path.dirname(__file__)
    raw_data_path = os.path.join(current_dir, '..', 'diabetes_prediction_dataset.csv')
    
    # Path untuk menyimpan data bersih di dalam preprocessing/
    cleaned_output_path = os.path.join(current_dir, 'diabetes_cleaned.csv') 

    print(f"Mencoba memproses dataset dari: {os.path.abspath(raw_data_path)}")
    print(f"Akan menyimpan data bersih ke: {os.path.abspath(cleaned_output_path)}")

    try:
        processed_df = preprocess_
