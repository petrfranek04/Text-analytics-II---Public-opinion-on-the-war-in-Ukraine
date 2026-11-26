"""Paths and constants shared by all scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # the project folder
DATA = ROOT / "data"
SAMPLE = DATA / "sample"        # sample, labels and split
RESULTS = DATA / "results"      # model predictions and scores
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"        # trained models
# Large training checkpoints stay outside the project, so cloud drives do not sync them
CHECKPOINTS = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ukraine_stance"

RAW_FILE = DATA / "reddit_raw_FINAL.csv"
CLEAN_FILE = DATA / "reddit_cleaned_full.csv"
if not CLEAN_FILE.exists():      # place used by 2data_cleaning.py
    CLEAN_FILE = DATA / "raw" / "reddit_cleaned_full.csv"

SEED = 42
N_BOOT = 1000                   # number of bootstrap samples for 95 % intervals

LABELS = ["pro", "against", "neutral", "na"]      # na = the comment is not about the country
COUNTRIES = {"russia": "Russia", "ukraine": "Ukraine"}
STANCE_COLUMNS = ["stance_russia", "stance_ukraine"]

# Keywords that show which country a comment mentions
UKR_PATTERN = r"ukrain|kyiv|kiev|zelensk|kharkiv|kherson|donbas|mariupol|bakhmut|azov|avdiivka|kramatorsk"
RUS_PATTERN = r"russia|putin|moscow|kremlin|ruzzia|vatnik|wagner|kursk|rosgvard|\bru\b"

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ZEROSHOT_MODEL = "facebook/bart-large-mnli"
BASE_MODEL = "roberta-base"
