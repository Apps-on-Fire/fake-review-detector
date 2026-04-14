import os
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("reviews.csv")

treino, teste = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

os.makedirs("treinamento", exist_ok=True)
os.makedirs("testes", exist_ok=True)

treino.to_csv("treinamento/reviews_treino.csv", index=False)
teste.to_csv("testes/reviews_teste.csv", index=False)

print(f"Treino: {len(treino)} registros")
print(f"Teste:  {len(teste)} registros")
print(f"Proporção label no treino:\n{treino['label'].value_counts()}")
print(f"Proporção label no teste:\n{teste['label'].value_counts()}")
