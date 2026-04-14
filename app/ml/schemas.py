from pydantic import BaseModel


class ClassificationResult(BaseModel):
    predicted_label: str  # "REAL" or "FAKE"
    confidence: float
    top_signals: list[str]
