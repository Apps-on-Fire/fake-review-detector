from langgraph.graph import StateGraph, START, END
from app.graph.state import ReviewAnalysisState
from app.graph.nodes.preprocess_node import preprocess_node
from app.graph.nodes.classify_node import classify_node
from app.graph.nodes.retrieve_node import retrieve_neighbors_node
from app.graph.nodes.explain_node import explain_node
from app.graph.nodes.format_node import format_response_node


def build_workflow() -> StateGraph:
    graph = StateGraph(ReviewAnalysisState)

    graph.add_node("preprocess", preprocess_node)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve_neighbors", retrieve_neighbors_node)
    graph.add_node("explain", explain_node)
    graph.add_node("format_response", format_response_node)

    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "classify")
    graph.add_edge("classify", "retrieve_neighbors")
    graph.add_edge("retrieve_neighbors", "explain")
    graph.add_edge("explain", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


review_workflow = build_workflow()
