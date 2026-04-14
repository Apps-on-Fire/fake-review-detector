"""
Script offline para treinar o classificador de reviews falsos.

Usa TF-IDF + LogisticRegression com Pipeline e ColumnTransformer
para combinar features de texto, categoria e rating.

Uso:
    python -m app.offline.train
"""

import os
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "treinamento", "reviews_treino.csv")
TEST_PATH = os.path.join(BASE_DIR, "testes", "reviews_teste.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "app", "ml", "artifacts")


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text_", "label"])
    df = df[df["label"].isin(["CG", "OR"])].copy()
    df["text_clean"] = df["text_"].apply(clean_text)
    return df


def build_pipeline() -> Pipeline:
    text_transformer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "text_clean"),
            ("category", OneHotEncoder(handle_unknown="ignore"), ["category"]),
            ("rating", StandardScaler(), ["rating"]),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
        )),
    ])
    return pipeline


def main():
    print("Carregando dados de treino...")
    train_df = load_and_prepare(TRAIN_PATH)
    print(f"  Treino: {len(train_df)} registros")
    print(f"  Distribuição: {train_df['label'].value_counts().to_dict()}")

    print("\nCarregando dados de teste...")
    test_df = load_and_prepare(TEST_PATH)
    print(f"  Teste: {len(test_df)} registros")

    X_train = train_df[["text_clean", "category", "rating"]]
    y_train = train_df["label"]
    X_test = test_df[["text_clean", "category", "rating"]]
    y_test = test_df["label"]

    print("\nTreinando pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("\nAvaliando no conjunto de teste...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["CG (Fake)", "OR (Real)"])

    print(f"  Acurácia: {accuracy:.4f}")
    print(f"\n{report}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    pipeline_path = os.path.join(ARTIFACTS_DIR, "classifier.joblib")
    joblib.dump(pipeline, pipeline_path)
    print(f"\nPipeline salvo em: {pipeline_path}")

    model_info = {
        "model_type": "TF-IDF + LogisticRegression",
        "accuracy": round(accuracy, 4),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "features": ["text_clean (TF-IDF)", "category (OneHot)", "rating (Scaled)"],
        "tfidf_max_features": 20000,
        "ngram_range": [1, 2],
    }
    info_path = os.path.join(ARTIFACTS_DIR, "model_info.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Metadados salvos em: {info_path}")


if __name__ == "__main__":
    main()
