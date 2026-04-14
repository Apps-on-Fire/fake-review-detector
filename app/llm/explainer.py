import os
from openai import OpenAI

SYSTEM_PROMPT = """\
You are an expert in fake review detection.

CRITICAL RULES:
- You do NOT decide the label. The official label has already been decided by the supervised classifier.
- Your role is ONLY to explain the decision using the provided signals and the retrieved similar reviews.
- Never contradict or change the provided label.

Your explanation must:
1. Summarize why the classifier reached this conclusion
2. Compare with the similar reviews found
3. Point out relevant patterns
4. Be clear and concise (3-5 sentences)

Respond in English.
"""


def explain_verdict(
    review_text: str,
    predicted_label: str,
    confidence: float,
    signals: list[str],
    similar_reviews: list[dict],
) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    similar_block = ""
    for i, sr in enumerate(similar_reviews[:3], 1):
        lbl = "FAKE" if sr["label"] == "CG" else "REAL"
        similar_block += (
            f"\n  {i}. [Score: {sr['score']}] [{lbl}] \"{sr['text'][:150]}...\""
        )

    verdict = "FAKE (computer generated)" if predicted_label == "FAKE" else "REAL (authentic)"

    user_msg = f"""\
Review analyzed:
\"{review_text}\"

Classifier decision: {verdict}
Confidence: {confidence:.1%}
Detected signals: {', '.join(signals) if signals else 'no strong signals'}

Similar reviews found in the database:{similar_block if similar_block else ' none found'}

Explain this decision clearly and objectively.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content
