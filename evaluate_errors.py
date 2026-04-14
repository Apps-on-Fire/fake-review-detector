"""
Evaluates the classifier on the test dataset and saves misclassified reviews to a CSV file.
"""

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from app.ml.classifier import ReviewClassifier
from app.ml.preprocessing import clean_text

TEST_PATH = "testes/reviews_teste.csv"
OUTPUT_PATH = "testes/misclassified_reviews.csv"

print("Loading test data...")
df = pd.read_csv(TEST_PATH)
df = df.dropna(subset=["text_", "label"])
df = df[df["label"].isin(["CG", "OR"])].copy()
print(f"  {len(df)} reviews to evaluate")

print("Loading classifier...")
clf = ReviewClassifier()

print("Running predictions...")
results = []
for i, row in df.iterrows():
    pred = clf.predict(row["text_"], row["category"], row["rating"])
    true_label = "FAKE" if row["label"] == "CG" else "REAL"
    if pred.predicted_label != true_label:
        results.append({
            "true_label": true_label,
            "predicted_label": pred.predicted_label,
            "confidence": pred.confidence,
            "signals": "; ".join(pred.top_signals),
            "category": row["category"],
            "rating": row["rating"],
            "text": row["text_"],
        })

errors_df = pd.DataFrame(results)
errors_df.to_csv(OUTPUT_PATH, index=False)

total = len(df)
wrong = len(errors_df)
accuracy = (total - wrong) / total

print(f"\nResults:")
print(f"  Total: {total}")
print(f"  Correct: {total - wrong}")
print(f"  Wrong: {wrong}")
print(f"  Accuracy: {accuracy:.2%}")
print(f"\nMisclassified reviews saved to: {OUTPUT_PATH}")
