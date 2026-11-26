# ============================================================
# BLOK 5: PŘEDPOVĚDI MODELŮ (VADER = baseline, RoBERTa = silnější model)
# Co dělá: načte ručně označený vzorek (400 komentářů) a ke každému
#   komentáři přidá předpověď sentimentu od dvou modelů:
#     - VADER (slovníkový): compound skóre, práh ±0.05 -> 3 třídy
#     - RoBERTa (cardiffnlp, trénovaná na sociálních sítích): 3 třídy
#       + pravděpodobnosti a spojité skóre (pozitivní minus negativní)
#   Modely hodnotí CELÝ text komentáře, ne sentiment vůči cíli
#   (Ukrajina/Rusko). Právě v tom rozdílu budou chyby pro error analysis.
# Vstup:  data/sample/gold_labels_v2.csv  (nahrajte na Drive předem)
# Výstup: data/sample/gold_with_predictions.csv
# Požadavky: zapnuté GPU (Runtime -> Change runtime type -> T4 GPU)
#            a nainstalované balíčky (vaderSentiment, transformers).
# ============================================================
import pandas as pd
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

BASE = "/content/drive/MyDrive/ukraine_project"
gold = pd.read_csv(f"{BASE}/data/sample/gold_labels_v2.csv")
texts = gold["text_sentiment"].tolist()

# --- VADER (baseline) ---
vader = SentimentIntensityAnalyzer()
gold["vader_compound"] = [vader.polarity_scores(t)["compound"] for t in texts]
gold["vader_label"] = gold["vader_compound"].apply(
    lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral"))

# --- RoBERTa (silnější model) ---
clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None,
    device=0 if torch.cuda.is_available() else -1,
)
res = clf(texts, batch_size=32, truncation=True, max_length=512)

for k in ["negative", "neutral", "positive"]:
    gold[f"rob_{k}"] = [{d["label"].lower(): d["score"] for d in r}[k] for r in res]

cols = ["rob_negative", "rob_neutral", "rob_positive"]
gold["roberta_label"] = gold[cols].idxmax(axis=1).str.replace("rob_", "")
gold["roberta_score"] = gold["rob_positive"] - gold["rob_negative"]   # spojité skóre -1..+1

gold.to_csv(f"{BASE}/data/sample/gold_with_predictions.csv", index=False)

# Rychlá kontrola, že vše proběhlo (zatím proti původnímu tónu)
print("VADER   shoda s tone_label:", (gold.vader_label == gold.tone_label).mean().round(3))
print("RoBERTa shoda s tone_label:", (gold.roberta_label == gold.tone_label).mean().round(3))
print(gold[["vader_label", "roberta_label"]].apply(pd.Series.value_counts))
