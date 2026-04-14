from app.graph.state import ReviewAnalysisState
from app.ml.preprocessing import clean_text


def preprocess_node(state: ReviewAnalysisState) -> dict:
    cleaned = clean_text(state["input_text"])
    return {"cleaned_text": cleaned}
