from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# 1. KONFIGURASI LOKASI
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "app" / "model_ai"
MODEL_PATH = MODEL_DIR / "spam_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "spam_vectorizer.joblib"

# =====================================================
# 2. MEMBACA DATASET
# =====================================================
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_PATH}\n"
            "Pastikan file SMSSpamCollection berada di folder data."
        )

    print("=" * 60)
    print("MEMBACA DATASET SMS SPAM COLLECTION")
    print("=" * 60)

    df = pd.read_csv(
        DATASET_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.strip()
    df["message"] = df["message"].astype(str).str.strip()

    df = df[
        df["label"].isin(["ham", "spam"])
        & df["message"].ne("")
    ].copy()

    if df.empty:
        raise ValueError("Dataset tidak memiliki data yang valid.")

    print(f"Jumlah data valid : {len(df)}")
    print("\nDistribusi label:")
    print(df["label"].value_counts())
    print("\nContoh data:")
    print(df.head())
    return df

# =====================================================
# 3. MEMBAGI DATA LATIH DAN DATA UJI
# =====================================================
def split_dataset(df):
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n" + "=" * 60)
    print("PEMBAGIAN DATASET")
    print("=" * 60)
    print(f"Data pelatihan : {len(X_train)}")
    print(f"Data pengujian : {len(X_test)}")
    return X_train, X_test, y_train, y_test

# =====================================================
# 4. MEMBANGUN DAN MELATIH MODEL
# =====================================================
def train_model(X_train, y_train):
    print("\n" + "=" * 60)
    print("MELATIH MODEL MACHINE LEARNING")
    print("=" * 60)

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.1)
        )
    ])

    model.fit(X_train, y_train)
    print("Pelatihan model selesai.")
    return model

# =====================================================
# 5. EVALUASI MODEL
# =====================================================
def evaluate_model(model, X_test, y_test):
    print("\n" + "=" * 60)
    print("EVALUASI MODEL")
    print("=" * 60)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Accuracy (%): {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=["ham", "spam"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print("Urutan label: ham, spam")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=["ham", "spam"]
        )
    )
    return accuracy

# =====================================================
# 6. MENYIMPAN MODEL DAN VECTORIZER
# =====================================================
def save_model(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Pipeline mencakup TF-IDF dan classifier.
    # Model dan vectorizer juga disimpan terpisah
    # sesuai kebutuhan integrasi aplikasi.

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\n" + "=" * 60)
    print("MODEL BERHASIL DISIMPAN")
    print("=" * 60)
    print(f"Pipeline model : {MODEL_PATH}")
    print(f"Vectorizer     : {VECTORIZER_PATH}")
    return classifier

# =====================================================
# 7. UJI PREDIKSI CONTOH
# =====================================================
def test_prediction(model):
    print("\n" + "=" * 60)
    print("UJI PREDIKSI SMS")
    print("=" * 60)

    samples = [
        "Congratulations! You have won a free prize. Call now!",
        "Are we still meeting for lunch today?",
        "URGENT! Claim your free cash reward now!"
    ]

    for message in samples:
        prediction = model.predict([message])[0]
        print(f"\nPesan     : {message}")
        print(f"Prediksi  : {prediction.upper()}")

# =====================================================
# 8. PROGRAM UTAMA
# =====================================================
def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    test_prediction(model)
    print("\nProses pengembangan model selesai.")

if __name__ == "__main__":
    main()
