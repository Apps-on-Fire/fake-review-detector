from typing import TypedDict


class ReviewAnalysisState(TypedDict, total=False):
    request_id: str
    input_text: str
    category: str
    rating: float

    cleaned_text: str

    classifier_label: str
    classifier_confidence: float
    classifier_signals: list[str]

    similar_reviews: list[dict]

    explanation: str

    final_output: dict
