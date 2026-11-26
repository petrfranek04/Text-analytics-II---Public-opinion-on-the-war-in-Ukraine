# ============================================================
# BLOK 12: ZERO-SHOT POSTOJ VŮČI CÍLI (NLI model)
# Co dělá:
#   Pro komentáře s cílem Ukrajina/Rusko pustí NLI model
#   (facebook/bart-large-mnli) s hypotézami o postoji autora vůči
#   konkrétní zemi, např. "The author's attitude toward Russia is
#   negative." Tím vznikne první model, který skutečně pracuje s
#   cílem, ne s tónem celého textu.
#   Dvě formulace hypotéz (P1 = hlavní, určená předem, P2 = kontrola
#   citlivosti na formulaci). Obě se počítají pro Rusko i Ukrajinu u
#   všech komentářů; cíl se pak vybere podle gold cíle.
#   Vyhodnocení: accuracy, macro-F1 (95% bootstrap), recall po
#   třídách, vedle VADER, RoBERTy a baselinů, na třech sadách:
#   náhodný vzorek, vše spojené (labely od Claude) a lidská sada.
# Vstup:  data/sample/all_gold_predictions.csv
#         data/sample/human_check_annotated.csv
# Výstup: data/sample/zeroshot_stance.csv, zeroshot_table.csv
# Požadavky: GPU (Runtime -> Change runtime type -> T4 GPU),
#   balíčky transformers (nic dalšího z dřívějších bloků).
# Pozor: výběr hypotézy používá gold cíl (horní odhad, jako v Bloku 8).
# ============================================================
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from transformers import pipeline
from google.colab import drive
drive.mount('/content/drive')

BASE = "/content/drive/MyDrive/ukraine_project"
SENT = ["negative", "neutral", "positive"]
rng = np.random.default_rng(42)
N_BOOT = 1000

pred = pd.read_csv(f"{BASE}/data/sample/all_gold_predictions.csv")
hum = pd.read_csv(f"{BASE}/data/sample/human_check_annotated.csv")
sub = pred[pred["target"].isin(["ukraine", "russia"])].copy().reset_index(drop=True)
print("Komentářů:", len(sub), "| GPU:", torch.cuda.is_available())

# ---------- 1) Zero-shot skóre ----------
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
              device=0 if torch.cuda.is_available() else -1)

PROMPTS = {
    "P1": ("The author's attitude toward {T} is {}.",
           {"negative": "negative", "neutral": "neutral", "positive": "positive"}),
    "P2": ("The author {} {T}.",
           {"negative": "criticizes or opposes", "neutral": "is neutral about",
            "positive": "supports or praises"}),
}
COUNTRIES = {"russia": "Russia", "ukraine": "Ukraine"}

def run(texts, tmpl, labmap):
    inv = {v: k for k, v in labmap.items()}
    out = zs(texts, candidate_labels=list(labmap.values()),
             hypothesis_template=tmpl, batch_size=8)
    rows = []
    for o in out:
        d = {inv[l]: s for l, s in zip(o["labels"], o["scores"])}
        rows.append([d["negative"], d["neutral"], d["positive"]])
    return np.array(rows)

texts = sub["text_sentiment"].tolist()
for p, (tmpl, labmap) in PROMPTS.items():
    for key, name in COUNTRIES.items():
        sc = run(texts, tmpl.replace("{T}", name), labmap)
        for j, s in enumerate(SENT):
            sub[f"{p}_{key}_{s}"] = sc[:, j]
        print(f"hotovo: {p} / {name}")

def pick(df, p, target_col):
    """Zero-shot label pro cíl uvedený ve sloupci target_col."""
    out = []
    for _, r in df.iterrows():
        sc = [r[f"{p}_{r[target_col]}_{s}"] for s in SENT]
        out.append(SENT[int(np.argmax(sc))])
    return out

sub.to_csv(f"{BASE}/data/sample/zeroshot_stance.csv", index=False)

# ---------- 2) Sady pro vyhodnocení ----------
def prepare(df, gold_col, target_col):
    d = df.copy()
    d["ZS_P1"] = pick(d, "P1", target_col)
    d["ZS_P2"] = pick(d, "P2", target_col)
    d["B0"] = "negative"
    d["B1"] = d[target_col].map({"russia": "negative", "ukraine": "positive"})
    d["gold"] = d[gold_col]
    return d

claude = prepare(sub, "target_sentiment", "target")
h = hum[["row_id", "target", "target_sentiment"]].rename(
    columns={"target": "target_h", "target_sentiment": "gold_h"})
hh = sub.merge(h, on="row_id")
hh = hh[hh["target_h"].isin(["ukraine", "russia"])].copy()
human = prepare(hh, "gold_h", "target_h")

SETS = {
    "Claude: náhodný": claude[claude["sample_type"] == "random"],
    "Claude: vše spojené": claude,
    "LIDÉ (těžší sada)": human,
}
MODELS = {"B0 vždy negativní": "B0", "B1 podle cíle": "B1", "VADER": "vader_label",
          "RoBERTa": "roberta_label", "Zero-shot P1": "ZS_P1", "Zero-shot P2": "ZS_P2"}

acc = lambda d, c: accuracy_score(d["gold"], d[c])
mf1 = lambda d, c: f1_score(d["gold"], d[c], labels=SENT, average="macro", zero_division=0)

def ci(fn, d, c):
    return np.percentile([fn(d.iloc[rng.integers(0, len(d), len(d))], c)
                          for _ in range(N_BOOT)], [2.5, 97.5])

rows = []
for sname, d in SETS.items():
    for mname, c in MODELS.items():
        al, ah = ci(acc, d, c); fl, fh = ci(mf1, d, c)
        rec = recall_score(d["gold"], d[c], labels=SENT, average=None, zero_division=0)
        rows.append({"sada": sname, "n": len(d), "model": mname,
                     "accuracy": round(acc(d, c), 3), "acc 95% CI": f"{al:.2f}-{ah:.2f}",
                     "macro-F1": round(mf1(d, c), 3), "F1 95% CI": f"{fl:.2f}-{fh:.2f}",
                     "rec neg": round(rec[0], 2), "rec neu": round(rec[1], 2),
                     "rec pos": round(rec[2], 2)})
res = pd.DataFrame(rows)
print("\n=== MODELY, BASELINY A ZERO-SHOT ===")
print(res.to_string(index=False))
res.to_csv(f"{BASE}/data/sample/zeroshot_table.csv", index=False)

# ---------- 3) Confusion matrix zero-shot P1 ----------
d = SETS["Claude: vše spojené"]
cm = pd.DataFrame(confusion_matrix(d["gold"], d["ZS_P1"], labels=SENT),
                  index=[f"gold_{s}" for s in SENT], columns=[f"pred_{s}" for s in SENT])
print("\nZero-shot P1 - confusion matrix (Claude, vše spojené):")
print(cm)

# ---------- 4) Spárovaný bootstrap: Zero-shot P1 - RoBERTa ----------
print("\n=== Spárovaný bootstrap: Zero-shot P1 - RoBERTa ===")
for sname, d in SETS.items():
    da, df_ = [], []
    for _ in range(N_BOOT):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        da.append(acc(s, "ZS_P1") - acc(s, "roberta_label"))
        df_.append(mf1(s, "ZS_P1") - mf1(s, "roberta_label"))
    a0 = acc(d, "ZS_P1") - acc(d, "roberta_label")
    f0 = mf1(d, "ZS_P1") - mf1(d, "roberta_label")
    print(f"{sname:20s} acc {a0:+.3f} [{np.percentile(da,2.5):+.3f}, {np.percentile(da,97.5):+.3f}] | "
          f"F1 {f0:+.3f} [{np.percentile(df_,2.5):+.3f}, {np.percentile(df_,97.5):+.3f}]")
