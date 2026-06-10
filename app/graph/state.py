from typing import TypedDict


class ReviewAnalysisState(TypedDict, total=False):
    request_id: str
    input_text: str
    category: str
    rating: float

    # Per-user OpenAI key, supplied by the caller and scoped to their MCP session.
    openai_api_key: str

    cleaned_text: str

    classifier_label: str
    classifier_confidence: float
    classifier_signals: list[str]

    similar_reviews: list[dict]

    explanation: str

    final_output: dict
