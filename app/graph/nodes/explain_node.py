from app.graph.state import ReviewAnalysisState
from app.llm.explainer import explain_verdict


def explain_node(state: ReviewAnalysisState) -> dict:
    explanation = explain_verdict(
        review_text=state["input_text"],
        predicted_label=state["classifier_label"],
        confidence=state["classifier_confidence"],
        signals=state["classifier_signals"],
        similar_reviews=state.get("similar_reviews", []),
    )
    return {"explanation": explanation}
