import os
import joblib
from app.ml.schemas import ClassificationResult
from app.ml.preprocessing import clean_text

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


class ReviewClassifier:
    def __init__(self):
        pipeline_path = os.path.join(ARTIFACTS_DIR, "classifier.joblib")
        self.pipeline = joblib.load(pipeline_path)
        self._feature_names = None

    def predict(self, text: str, category: str, rating: float) -> ClassificationResult:
        import pandas as pd

        cleaned = clean_text(text)
        df = pd.DataFrame([{
            "text_clean": cleaned,
            "category": category,
            "rating": rating,
        }])

        proba = self.pipeline.predict_proba(df)[0]
        classes = self.pipeline.classes_
        pred_idx = proba.argmax()
        raw_label = classes[pred_idx]
        confidence = float(proba[pred_idx])

        label = "FAKE" if raw_label == "CG" else "REAL"
        signals = self._extract_signals(text, cleaned, rating, confidence)

        return ClassificationResult(
            predicted_label=label,
            confidence=round(confidence, 4),
            top_signals=signals,
        )

    def _extract_signals(
        self, raw_text: str, cleaned: str, rating: float, confidence: float
    ) -> list[str]:
        signals = []
        words = cleaned.split()

        if len(words) < 10:
            signals.append("texto muito curto")

        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < 0.5:
            signals.append("alta repetição de palavras")

        exclamations = raw_text.count("!")
        if exclamations >= 3:
            signals.append("excesso de exclamações")

        if len(words) < 15 and rating == 5.0:
            signals.append("review curto com nota máxima")

        generic_phrases = [
            "love it", "great product", "highly recommend",
            "works great", "love this", "best ever",
        ]
        lower_raw = raw_text.lower()
        found = [p for p in generic_phrases if p in lower_raw]
        if len(found) >= 2:
            signals.append("linguagem excessivamente genérica")

        if confidence >= 0.85:
            signals.append("alta confiança do classificador")
        elif confidence < 0.6:
            signals.append("baixa confiança — caso limítrofe")

        return signals[:5]
