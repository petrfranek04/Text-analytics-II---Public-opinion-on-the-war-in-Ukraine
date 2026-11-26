import re

def clean_scraped_data(df):
    """
    Vyčistí surová data z Redditu a zaznamená, kolik řádků vypadlo v každém kroku.
    Vytvoří dva textové sloupce:
      - 'text_sentiment': ponechá interpunkci (!!!) a velká písmena pro VADER a RoBERTa.
      - 'text_topic': odstraní markdown a odkazy pro čisté shlukování v BERTopic.
    """
    # Slovník pro sledování počtu záznamů v jednotlivých fázích čištění (do reportu)
    stats = {"1_Hruba_data_celkem": len(df)}

    # 1. Odstranění duplicitních ID (pokud se stejný příspěvek stáhl vícekrát)
    df = df.drop_duplicates(subset=["record_id"], keep="first").copy()
    stats["2_Po_odstraneni_duplicitnich_ID"] = len(df)

    # 2. Odstranění smazaných komentářů ([deleted], [removed]) a automatických botů
    df = df[~df["author_hash"].isin(["deleted_or_bot"])]
    df = df[~df["text"].isin(["[removed]", "[deleted]", ""])]

    # Odfiltrujeme typické hlášky moderátorských botů v textu
    bot_patterns = r"I am a bot|performed automatically|RemindMe!|submission statement"
    df = df[~df["text"].str.contains(bot_patterns, case=False, na=False)]
    stats["3_Po_odstraneni_botu_a_smazanych"] = len(df)

    # 3. Odstranění citací (řádky začínající '>') a URL odkazů.
    # DŮLEŽITÉ: Na Redditu lidé citují názor oponenta pomocí '>' a pak s ním nesouhlasí.
    # Kdybychom citaci nechali, model sentimentu by hodnotil slova oponenta!
    def clean_for_sentiment(text):
        text = re.sub(r"(?m)^>.*$", "", str(text))        # Smaže citované řádky
        text = re.sub(r"http\S+|www\.\S+", "", text)      # Smaže webové odkazy
        return re.sub(r"\s+", " ", text).strip()          # Sjednotí mezery

    df["text_sentiment"] = df["text"].apply(clean_for_sentiment)

    # 4. Silnější čištění pro BERTopic (odstraní odkazy na subreddity r/..., uživatele u/... a hvězdičky)
    def clean_for_bertopic(text):
        text = re.sub(r"/?[ru]/[A-Za-z0-9_-]+", "", text)
        text = re.sub(r"[\*\[\]\(\)~_`#]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    df["text_topic"] = df["text_sentiment"].apply(clean_for_bertopic)

    # 5. Odstranění spamu (identické texty kopírované do více vláken)
    df = df.drop_duplicates(subset=["text_sentiment"], keep="first")
    stats["4_Po_odstraneni_spamu"] = len(df)

    # 6. Odstranění příliš krátkých výkřiků (méně než 5 slov, např. "Agreed!", "This.")
    df["word_count"] = df["text_sentiment"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= 5].copy()
    stats["5_Finalni_cista_data"] = len(df)

    return df, pd.Series(stats, name="Pocet_zaznamu")

# Spustíme čištění na stažených datech
df_clean, funnel_table = clean_scraped_data(df_raw_all)

print("=== PŘEHLED ČIŠTĚNÍ DAT (Uložte si do report.pdf!) ===")
display(funnel_table.to_frame())

# Uložíme kompletní čistý dataset pro vaše kolegy v týmu (pro modely VADER, RoBERTa, BERTopic)
df_clean.to_csv("data/raw/reddit_cleaned_full.csv", index=False)

# Uložíme malý anonymizovaný vzorek (100 řádků) bez surového textu, který smíte nahrát na GitHub
os.makedirs("data/sample", exist_ok=True)
df_sample = df_clean.drop(columns=["text"]).sample(n=min(100, len(df_clean)), random_state=42)
df_sample.to_csv("data/sample/sample_anonymized.csv", index=False)

print(f"\nČistý dataset uložen do 'data/raw/reddit_cleaned_full.csv' ({len(df_clean)} řádků).")
print("Anonymizovaný vzorek (100 řádků) pro GitHub uložen do 'data/sample/sample_anonymized.csv'.")
display(df_clean[["event", "subreddit", "type", "word_count", "text_sentiment"]].head(10))
