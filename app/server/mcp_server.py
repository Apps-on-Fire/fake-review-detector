import os
import sys
import json
import uuid

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mcp.server.fastmcp import FastMCP
from app.graph.workflow import review_workflow

AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "")

mcp = FastMCP(
    "Fake Review Detector",
    instructions=(
        "MCP server for fake review detection. "
        "Uses a supervised classifier (TF-IDF + LogisticRegression) to decide, "
        "Pinecone to retrieve similar reviews, and an LLM to explain the decision."
    ),
    host="0.0.0.0",
    port=7860,
)


def _check_auth(token: str) -> bool:
    if not AUTH_TOKEN:
        return True
    return token == AUTH_TOKEN


@mcp.tool()
async def analyze_review(text: str, category: str, rating: float, auth_token: str = "") -> str:
    """Analyzes a review and returns the classification (FAKE/REAL) with explanation.

    The supervised classifier decides the label.
    Pinecone retrieves similar reviews.
    The LLM explains the decision.

    Args:
        text: Review text to analyze
        category: Product category (e.g. Home_and_Kitchen_5)
        rating: Review rating (1.0 to 5.0)
        auth_token: Authentication token (required for remote access)
    """
    if not _check_auth(auth_token):
        return json.dumps({"error": "Unauthorized. Invalid auth_token."})

    state = {
        "request_id": str(uuid.uuid4()),
        "input_text": text,
        "category": category,
        "rating": rating,
    }

    result = await review_workflow.ainvoke(state)
    return json.dumps(result["final_output"], ensure_ascii=False, indent=2)


@mcp.tool()
async def get_similar_reviews(text: str, category: str = "", top_k: int = 5, auth_token: str = "") -> str:
    """Searches for similar reviews in Pinecone and returns pattern analysis.

    Retrieves the most similar reviews from the vector database and uses the LLM
    to analyze patterns among them.

    Args:
        text: Review text to search for similar ones
        category: Category to filter by (optional)
        top_k: Number of similar reviews to return (default 5)
        auth_token: Authentication token (required for remote access)
    """
    if not _check_auth(auth_token):
        return json.dumps({"error": "Unauthorized. Invalid auth_token."})

    from app.retrieval.pinecone_client import query_similar
    from app.llm.explainer import explain_verdict

    similar = query_similar(text=text, category=category or None, top_k=top_k)

    fake_count = sum(1 for s in similar if s["label"] == "CG")
    real_count = sum(1 for s in similar if s["label"] == "OR")

    explanation = explain_verdict(
        review_text=text,
        predicted_label="FAKE" if fake_count > real_count else "REAL",
        confidence=fake_count / max(len(similar), 1),
        signals=[
            f"{fake_count}/{len(similar)} neighbors are fake",
            f"{real_count}/{len(similar)} neighbors are real",
        ],
        similar_reviews=similar,
    )

    output = {
        "query_text": text,
        "similar_reviews": similar,
        "pattern_analysis": explanation,
        "neighbor_stats": {
            "total": len(similar),
            "fake_count": fake_count,
            "real_count": real_count,
        },
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run()
