from google.colab import drive
import pandas as pd
import os

drive.mount('/content/drive')
BASE = "/content/drive/MyDrive/ukraine_project"
os.makedirs(f"{BASE}/data/sample", exist_ok=True)

SEED = 42
N_PER_EVENT = 60        # 5 událostí x 60 = 300 náhodných komentářů
N_HARD = 100            # komentáře s náznakem sarkasmu (jen pro error analysis)

df = pd.read_csv(f"{BASE}/data/raw/reddit_cleaned_full_FINAL.csv").reset_index(drop=True)
df["row_id"] = df.index          # stabilní ID, podle něj spojíme výsledky
df = df.dropna(subset=["text_sentiment"])
print("Celkem řádků:", len(df))
print(df["event"].value_counts())

# Kandidáti: 5-150 slov, ať jsou komentáře čitelné při anotaci (omezení -> do reportu)
cand = df[(df["word_count"] >= 5) & (df["word_count"] <= 150)]

# 1) Náhodný stratifikovaný vzorek podle události
main = cand.groupby("event").sample(n=N_PER_EVENT, random_state=SEED).copy()
main["sample_type"] = "random"

# 2) Těžký vzorek: signály sarkasmu
sarc_pattern = (r"(/s\b|yeah,? right|sure,? buddy|oh,? great|great job|totally|"
                r"how (brave|original|surprising)|what a surprise|congrats|genius|"
                r"thanks,? (obama|biden|trump)|so much for)")
rest = cand[~cand["row_id"].isin(main["row_id"])]
hard_pool = rest[rest["text_sentiment"].str.contains(sarc_pattern, case=False, regex=True)]
print("Dostupných sarkastických kandidátů:", len(hard_pool))
hard = hard_pool.sample(n=min(N_HARD, len(hard_pool)), random_state=SEED).copy()
hard["sample_type"] = "sarcasm_hint"

sample = pd.concat([main, hard]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# Plná verze s metadaty (zůstává na Drive)
sample[["row_id", "event", "subreddit", "sample_type", "text_sentiment"]] \
    .to_csv(f"{BASE}/data/sample/sample_full.csv", index=False)

# Verze k anotaci: jen ID a text
sample[["row_id", "text_sentiment"]].assign(gold_label="") \
    .to_csv(f"{BASE}/data/sample/sample_to_label.csv", index=False)

print("Vzorek:", sample["sample_type"].value_counts().to_dict())

from google.colab import files
files.download(f"{BASE}/data/sample/sample_to_label.csv")


# ============================================================
# BLOK 4: CÍLENÝ VZOREK (komentáře zmiňující Ukrajinu)
# Co dělá: v náhodném vzorku bylo o Ukrajině jen 30 komentářů, což na
#   metriky nestačí. Tento blok vybere dalších ~150 komentářů, které
#   obsahují klíčová slova o Ukrajině (Ukrain*, Kyiv, Zelensky...),
#   rovnoměrně z každé události, a vyloučí ty, které už ve vzorku jsou.
# Vstup:  data/raw/reddit_cleaned_full_FINAL.csv, data/sample/sample_full.csv
# Výstup: data/sample/sample_to_label_2.csv  (tento soubor pošlete Claudovi)
#         data/sample/sample_targeted_full.csv (metadata, zůstává na Drive)
# Pozor:  tento vzorek NENÍ náhodný, metriky z něj se uvádějí zvlášť
#         a nesmí se použít k odhadu, jak často se o Ukrajině mluví.
# ============================================================
import pandas as pd

BASE = "/content/drive/MyDrive/ukraine_project"
SEED = 42
N_TARGETED = 150

df = pd.read_csv(f"{BASE}/data/raw/reddit_cleaned_full_FINAL.csv").reset_index(drop=True)
df["row_id"] = df.index
df = df.dropna(subset=["text_sentiment"])

already = pd.read_csv(f"{BASE}/data/sample/sample_full.csv")["row_id"]
cand = df[(df["word_count"].between(5, 150)) & (~df["row_id"].isin(already))]

ua_pattern = r"(ukrain|kyiv|kiev|zelensk|kharkiv|kherson|donbas|mariupol|bakhmut|azov)"
pool = cand[cand["text_sentiment"].str.contains(ua_pattern, case=False, regex=True)]
print("Kandidátů zmiňujících Ukrajinu:", len(pool))

per_event = N_TARGETED // pool["event"].nunique()
targeted = pool.groupby("event").sample(n=per_event, random_state=SEED)
targeted = targeted.sample(frac=1, random_state=SEED).reset_index(drop=True)

targeted.assign(sample_type="targeted_ua")[
    ["row_id", "event", "subreddit", "sample_type", "text_sentiment"]
].to_csv(f"{BASE}/data/sample/sample_targeted_full.csv", index=False)

targeted[["row_id", "text_sentiment"]].to_csv(
    f"{BASE}/data/sample/sample_to_label_2.csv", index=False)

from google.colab import files
files.download(f"{BASE}/data/sample/sample_to_label_2.csv")
print("Vzorek:", len(targeted))
