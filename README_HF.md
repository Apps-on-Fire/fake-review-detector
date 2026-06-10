---
title: Fake Review Detector MCP
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
---

# Fake Review Detector - MCP Server

MCP server for fake review detection using a hybrid architecture:
- **Supervised classifier** (TF-IDF + LogisticRegression) decides the label
- **Pinecone** retrieves similar reviews for context
- **LLM** (GPT-4o-mini) explains the decision
- **LangGraph** orchestrates the workflow

## Tools

- `analyze_review(text, category, rating, openai_api_key)` - Full classification pipeline
- `get_similar_reviews(text, category, top_k, openai_api_key)` - Similarity search + analysis

## Bring your own OpenAI key

This server does **not** use the host's OpenAI credits. On your **first tool call** in a
conversation, pass your own key as the `openai_api_key` argument (e.g.
`openai_api_key="sk-..."`). It is kept in memory **for that MCP session only** and reused
on later calls, so you don't resend it. Without a valid key the tools return a
`missing_openai_key` error — they never fall back to the host's key. Get a key at
<https://platform.openai.com/api-keys>.
