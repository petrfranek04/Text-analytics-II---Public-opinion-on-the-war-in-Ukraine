# ============================================================
# BLOK 11: LIDSKÁ VALIDACE (konsenzus 2 lidí vs. labely od Claude)
# Co dělá:
#   1) spočítá shodu lidského konsenzu s labely od Claude
#      (accuracy, Cohenova kappa) pro cíl i sentiment vůči cíli
#   2) ukáže shodu zvlášť pro labely, které Claude označil jako
#      jisté / nejisté (target_uncertain)
#   3) vyhodnotí VADER, RoBERTu a baseliny proti LIDSKÝM labelům
#      (komentáře, kde lidé určili cíl Ukrajina/Rusko) a vedle toho
#      proti labelům od Claude na stejných řádcích
#   4) uloží tabulku neshod pro failure log
# Vstup:  data/sample/all_gold_predictions.csv
#         data/sample/human_check_annotated.csv (nahrát na Drive)
# Výstup: data/sample/human_vs_claude_disagreements.csv
# Pozor:  lidé anotovali společně (konsenzus), takže jde o shodu
#   člověk-AI, ne o shodu mezi anotátory. Kontrolní sada obsahuje
#   21 záměrně nejednoznačných komentářů, shoda je proto spíš
#   podhodnocená oproti celému vzorku.
# ============================================================
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from google.colab import drive
drive.mount('/content/drive')

BASE = "/content/drive/MyDrive/ukraine_project"
SENT = ["negative", "neutral", "positive"]
rng = np.random.default_rng(42)

pred = pd.read_csv(f"{BASE}/data/sample/all_gold_predictions.csv")
hum = pd.read_csv(f"{BASE}/data/sample/human_check_annotated.csv")

cols_c = ["row_id", "text_sentiment", "target", "target_sentiment", "intensity",
          "target_uncertain", "vader_label", "roberta_label"]
m = hum[["row_id", "target", "target_sentiment", "intensity", "target_uncertain"]].merge(
    pred[cols_c], on="row_id", suffixes=("_h", "_c"))
print("Spojeno řádků:", len(m), "z", len(hum))

# ---------- 1) Shoda člověk x Claude ----------
def agree(a, b):
    return accuracy_score(a, b), cohen_kappa_score(a, b)

ta, tk = agree(m["target_h"], m["target_c"])
sa, sk = agree(m["target_sentiment_h"], m["target_sentiment_c"])
print(f"\nCÍL:       accuracy {ta:.3f} | kappa {tk:.3f}")
print(f"SENTIMENT: accuracy {sa:.3f} | kappa {sk:.3f}  (všech {len(m)}, 'na' = bez cíle)")
print("\nCíl: řádky = člověk, sloupce = Claude")
print(pd.crosstab(m["target_h"], m["target_c"]))
print("\nSentiment: řádky = člověk, sloupce = Claude")
print(pd.crosstab(m["target_sentiment_h"], m["target_sentiment_c"]))

# ---------- 2) Shoda podle jistoty labelu u Claude ----------
print("\n=== Shoda podle toho, zda Claude label označil jako nejistý ===")
for u, d in m.groupby("target_uncertain_c"):
    print(f"nejistý={u}  n={len(d):3d}  shoda cíle {(d.target_h == d.target_c).mean():.2f}  "
          f"shoda sentimentu {(d.target_sentiment_h == d.target_sentiment_c).mean():.2f}")

# ---------- 3) Modely proti lidským labelům ----------
h = m[m["target_h"].isin(["ukraine", "russia"])].copy()
h["B0_vzdy_negativni"] = "negative"
h["B1_podle_cile"] = h["target_h"].map({"russia": "negative", "ukraine": "positive"})
print(f"\nKomentářů s lidským cílem Ukrajina/Rusko: {len(h)}")

def boot(fn, d, n=1000):
    v = [fn(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])

rows = []
for gold_name, gold_col in [("LIDÉ", "target_sentiment_h"), ("Claude (stejné řádky)", "target_sentiment_c")]:
    for model, col in [("B0 vždy negativní", "B0_vzdy_negativni"), ("B1 podle cíle", "B1_podle_cile"),
                       ("VADER", "vader_label"), ("RoBERTa", "roberta_label")]:
        a = lambda d: accuracy_score(d[gold_col], d[col])
        f = lambda d: f1_score(d[gold_col], d[col], labels=SENT, average="macro", zero_division=0)
        al, ah = boot(a, h); fl, fh = boot(f, h)
        rows.append({"gold": gold_name, "model": model, "n": len(h),
                     "accuracy": round(a(h), 3), "acc 95% CI": f"{al:.2f}-{ah:.2f}",
                     "macro-F1": round(f(h), 3), "F1 95% CI": f"{fl:.2f}-{fh:.2f}"})
print("\n=== MODELY PROTI LIDSKÝM A CLAUDOVÝM LABELŮM ===")
print(pd.DataFrame(rows).to_string(index=False))

# ---------- 4) Tabulka neshod ----------
dis = m[(m.target_h != m.target_c) | (m.target_sentiment_h != m.target_sentiment_c)]
dis = dis[["row_id", "text_sentiment", "target_h", "target_c", "target_sentiment_h",
           "target_sentiment_c", "target_uncertain_c", "roberta_label"]]
dis.to_csv(f"{BASE}/data/sample/human_vs_claude_disagreements.csv", index=False)
print(f"\nNeshod celkem: {len(dis)} -> uloženo human_vs_claude_disagreements.csv")
