"""
Test script to run the MCP server tools directly without a client.
Usage: python test_mcp.py
"""

import asyncio
import json
import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from app.graph.workflow import review_workflow
from app.retrieval.pinecone_client import query_similar
from app.llm.explainer import explain_verdict


async def test_analyze_review():
    print("=" * 60)
    print("TEST 1: analyze_review (full workflow)")
    print("=" * 60)

    state = {
        "request_id": "test-001",
        "input_text": "Love this! Well made, sturdy, and very comfortable. Love it! Very pretty",
        "category": "Home_and_Kitchen_5",
        "rating": 5.0,
    }

    print(f"\nInput: {state['input_text'][:80]}...")
    print(f"Category: {state['category']}, Rating: {state['rating']}")
    print("\nRunning workflow...")

    result = await review_workflow.ainvoke(state)
    output = result["final_output"]

    print(f"\nLabel: {output['predicted_label']}")
    print(f"Confidence: {output['confidence']}")
    print(f"Signals: {output['signals']}")
    print(f"\nSimilar reviews found: {len(output['similar_reviews'])}")
    for sr in output["similar_reviews"][:3]:
        print(f"  - [{sr['label']}] score={sr['score']} \"{sr['text'][:80]}...\"")
    print(f"\nExplanation:\n{output['explanation']}")


async def test_get_similar_reviews():
    print("\n" + "=" * 60)
    print("TEST 2: get_similar_reviews (Pinecone retrieval + analysis)")
    print("=" * 60)

    text = "I have been using this product for 3 months now. The handle broke after the first week but customer service replaced it."
    print(f"\nQuery: {text[:80]}...")
    print("\nSearching Pinecone...")

    similar = query_similar(text=text, category=None, top_k=5)

    fake_count = sum(1 for s in similar if s["label"] == "CG")
    real_count = sum(1 for s in similar if s["label"] == "OR")

    print(f"\nResults: {len(similar)} similar reviews ({fake_count} fake, {real_count} real)")
    for sr in similar:
        lbl = "FAKE" if sr["label"] == "CG" else "REAL"
        print(f"  - [{lbl}] score={sr['score']} \"{sr['text'][:80]}...\"")

    print("\nGenerating analysis...")
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
    print(f"\nAnalysis:\n{explanation}")


async def main():
    print("Fake Review Detector - MCP Tools Test")
    print("=" * 60)

    await test_analyze_review()
    await test_get_similar_reviews()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
