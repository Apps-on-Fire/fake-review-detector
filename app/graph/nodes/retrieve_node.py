from app.graph.state import ReviewAnalysisState
from app.retrieval.pinecone_client import query_similar


def retrieve_neighbors_node(state: ReviewAnalysisState) -> dict:
    similar = query_similar(
        text=state["input_text"],
        category=state.get("category"),
        top_k=5,
    )
    return {"similar_reviews": similar}
