from app.graph.state import ReviewAnalysisState


def format_response_node(state: ReviewAnalysisState) -> dict:
    output = {
        "predicted_label": state["classifier_label"],
        "confidence": state["classifier_confidence"],
        "signals": state["classifier_signals"],
        "similar_reviews": state.get("similar_reviews", []),
        "explanation": state.get("explanation", ""),
    }
    return {"final_output": output}
