import os
import sys
import json
import uuid
import weakref

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from mcp.server.fastmcp import FastMCP, Context
from app.graph.workflow import review_workflow

AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "")

# When true, the server may fall back to its own OPENAI_API_KEY if the caller did
# not supply one. OFF by default so the public deployment is strictly bring-your-own
# key (callers never spend the deployer's OpenAI credits). Enable only for local dev.
ALLOW_SERVER_OPENAI_KEY = os.environ.get("ALLOW_SERVER_OPENAI_KEY", "").lower() in (
    "1",
    "true",
    "yes",
)

mcp = FastMCP(
    "Fake Review Detector",
    instructions=(
        "MCP server for fake review detection. "
        "Uses a supervised classifier (TF-IDF + LogisticRegression) to decide, "
        "Pinecone to retrieve similar reviews, and an LLM to explain the decision. "
        "On your first tool call in a conversation, pass your own OpenAI API key as "
        "the 'openai_api_key' argument; it is reused for the rest of the session."
    ),
    host="0.0.0.0",
    port=7860,
)

# Per-session cache of each caller's OpenAI key. Keyed by the MCP session object via a
# WeakKeyDictionary, so the entry is dropped automatically when the connection (i.e.
# the conversation) closes — no manual cleanup and no cross-session leakage.
_session_openai_keys: "weakref.WeakKeyDictionary[object, str]" = weakref.WeakKeyDictionary()

MISSING_KEY_MESSAGE = (
    "OpenAI API key required. On your first call in this conversation, pass your key "
    "as the 'openai_api_key' argument (e.g. openai_api_key='sk-...'). It is kept only "
    "in memory for this session and reused on later calls, so you do not need to send "
    "it again. Get a key at https://platform.openai.com/api-keys."
)

LANDING_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fake Review Detector — MCP Server</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 4rem auto;
           padding: 0 1.25rem; line-height: 1.55; color: #1a1a1a; }
    code { background: #f0f0f3; padding: .15rem .4rem; border-radius: 4px; }
    .tag { display: inline-block; background: #e8f0fe; color: #1558d6;
           border-radius: 999px; padding: .2rem .7rem; font-size: .8rem; }
    a { color: #1558d6; }
  </style>
</head>
<body>
  <span class="tag">MCP server · online</span>
  <h1>🔍 Fake Review Detector</h1>
  <p>This is a <strong>Model Context Protocol</strong> server, not a web app. Connect an
     MCP client (Claude.ai, VS Code, …) to the SSE endpoint:</p>
  <p><code>/sse</code></p>
  <h3>Bring your own OpenAI key</h3>
  <p>On your <strong>first tool call</strong> in a conversation, pass your own OpenAI API
     key as the <code>openai_api_key</code> argument. The server keeps it in memory only
     for that session and reuses it on later calls — you never spend anyone else's
     credits, and you don't resend the key each time.</p>
  <h3>Tools</h3>
  <ul>
    <li><code>analyze_review(text, category, rating, openai_api_key)</code></li>
    <li><code>get_similar_reviews(text, category, top_k, openai_api_key)</code></li>
  </ul>
  <p>Health check: <a href="/health">/health</a></p>
</body>
</html>
"""


@mcp.custom_route("/", methods=["GET"])
async def landing(request: Request) -> HTMLResponse:
    """Human-facing landing page so the container root returns 200 (not 404)."""
    return HTMLResponse(LANDING_HTML)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Lightweight health/liveness probe."""
    return JSONResponse(
        {
            "status": "ok",
            "service": "fake-review-detector-mcp",
            "mcp_endpoint": "/sse",
        }
    )


def _check_auth(token: str) -> bool:
    if not AUTH_TOKEN:
        return True
    return token == AUTH_TOKEN


def _get_session(ctx: Context):
    """Return the current MCP session object, or None if unavailable."""
    try:
        return ctx.session
    except Exception:
        return None


def _resolve_openai_key(ctx: Context, provided: str) -> str | None:
    """Resolve the OpenAI key to use for this call.

    Priority: a key passed on this call (also cached for the session) > a key cached
    earlier in this same session > the server's own key (only if explicitly allowed
    via ALLOW_SERVER_OPENAI_KEY). Returns None when none is available, so the caller
    can return an instructive error instead of falling back to the deployer's key.
    """
    session = _get_session(ctx)
    provided = (provided or "").strip()

    if provided:
        if session is not None:
            _session_openai_keys[session] = provided
        return provided

    if session is not None:
        cached = _session_openai_keys.get(session)
        if cached:
            return cached

    if ALLOW_SERVER_OPENAI_KEY:
        return os.environ.get("OPENAI_API_KEY") or None

    return None


@mcp.tool()
async def analyze_review(
    ctx: Context,
    text: str,
    category: str,
    rating: float,
    openai_api_key: str = "",
    auth_token: str = "",
) -> str:
    """Analyzes a review and returns the classification (FAKE/REAL) with explanation.

    The supervised classifier decides the label.
    Pinecone retrieves similar reviews.
    The LLM explains the decision.

    Args:
        text: Review text to analyze
        category: Product category (e.g. Home_and_Kitchen_5)
        rating: Review rating (1.0 to 5.0)
        openai_api_key: Your OpenAI API key. Required on the first call of a
            conversation; reused automatically on later calls in the same session.
        auth_token: Authentication token (required only if the server sets MCP_AUTH_TOKEN)
    """
    if not _check_auth(auth_token):
        return json.dumps({"error": "Unauthorized. Invalid auth_token."})

    user_key = _resolve_openai_key(ctx, openai_api_key)
    if not user_key:
        return json.dumps({"error": "missing_openai_key", "message": MISSING_KEY_MESSAGE})

    state = {
        "request_id": str(uuid.uuid4()),
        "input_text": text,
        "category": category,
        "rating": rating,
        "openai_api_key": user_key,
    }

    result = await review_workflow.ainvoke(state)
    return json.dumps(result["final_output"], ensure_ascii=False, indent=2)


@mcp.tool()
async def get_similar_reviews(
    ctx: Context,
    text: str,
    category: str = "",
    top_k: int = 5,
    openai_api_key: str = "",
    auth_token: str = "",
) -> str:
    """Searches for similar reviews in Pinecone and returns pattern analysis.

    Retrieves the most similar reviews from the vector database and uses the LLM
    to analyze patterns among them.

    Args:
        text: Review text to search for similar ones
        category: Category to filter by (optional)
        top_k: Number of similar reviews to return (default 5)
        openai_api_key: Your OpenAI API key. Required on the first call of a
            conversation; reused automatically on later calls in the same session.
        auth_token: Authentication token (required only if the server sets MCP_AUTH_TOKEN)
    """
    if not _check_auth(auth_token):
        return json.dumps({"error": "Unauthorized. Invalid auth_token."})

    user_key = _resolve_openai_key(ctx, openai_api_key)
    if not user_key:
        return json.dumps({"error": "missing_openai_key", "message": MISSING_KEY_MESSAGE})

    from app.retrieval.pinecone_client import query_similar
    from app.llm.explainer import explain_verdict

    similar = query_similar(
        text=text, category=category or None, top_k=top_k, openai_api_key=user_key
    )

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
        openai_api_key=user_key,
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
