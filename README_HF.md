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

- `analyze_review(text, category, rating)` - Full classification pipeline
- `get_similar_reviews(text, category, top_k)` - Similarity search + analysis
