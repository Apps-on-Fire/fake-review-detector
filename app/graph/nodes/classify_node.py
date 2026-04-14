from app.graph.state import ReviewAnalysisState
from app.ml.classifier import ReviewClassifier

_classifier: ReviewClassifier | None = None


def _get_classifier() -> ReviewClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ReviewClassifier()
    return _classifier


def classify_node(state: ReviewAnalysisState) -> dict:
    clf = _get_classifier()
    result = clf.predict(
        text=state["input_text"],
        category=state["category"],
        rating=state["rating"],
    )
    return {
        "classifier_label": result.predicted_label,
        "classifier_confidence": result.confidence,
        "classifier_signals": result.top_signals,
    }
