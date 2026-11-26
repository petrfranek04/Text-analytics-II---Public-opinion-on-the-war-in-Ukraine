"""
Preparation step 1: download posts and comments about five war events from five subreddits
(Arctic Shift API, an archive of Reddit).
Output: data/reddit_raw_FINAL.csv
The data is already downloaded: if the raw or cleaned file exists, nothing happens
(a new download would change the rows that the labels are tied to).
"""
import sys
from pathlib import Path
import hashlib
import re
import time

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import CLEAN_FILE, RAW_FILE

# Five events. Time window as Unix timestamps (UTC), about four weeks around each event.
# "q" = keywords of the event (a thread matches if any of them appears).
EVENTS = {
    "1_invasion": {"after": 1645056000, "before": 1647561600, "q": "Ukraine|Russia|invasion|Kyiv"},
    "2_bucha": {"after": 1648166400, "before": 1650672000, "q": "Bucha|massacre|Irpin|war crimes"},
    "3_nordstream": {"after": 1663545600, "before": 1666051200, "q": "Nord Stream|pipeline|sabotage"},
    "4_grain": {"after": 1694131200, "before": 1696636800, "q": "Poland|grain|farmers|embargo"},
    "5_carlson_aid": {"after": 1706745600, "before": 1709251200, "q": "Tucker|Carlson|aid|funding|Putin"},
}

# Wanted number of records per subreddit for one event
QUOTAS = {"worldnews": 2500, "europe": 1900, "UkraineRussiaReport": 1900,
          "Conservative": 650, "politics": 650}

API = "https://arctic-shift.photon-reddit.com/api"
HEADERS = {"User-Agent": "UkraineOpinionStudy/1.0 (Academic Research Project)"}


def anonymize_user(author):
    """Replace a user name with a 12-character hash: same user = same code, name not stored."""
    if not author or str(author) in ["[deleted]", "AutoModerator", "None"]:
        return "deleted_or_bot"
    return hashlib.sha256(str(author).encode("utf-8")).hexdigest()[:12]


def quotas_for_event(event):
    """r/UkraineRussiaReport did not exist during the invasion, so its share
    goes to r/worldnews and r/europe."""
    quotas = QUOTAS.copy()
    if event == "1_invasion":
        share = quotas["UkraineRussiaReport"]
        quotas["UkraineRussiaReport"] = 0
        quotas["worldnews"] += share // 2
        quotas["europe"] += share - share // 2
    return quotas


def api_get(kind, params, max_tries=5):
    """One request to the API. Waits and tries again if the server is busy."""
    for attempt in range(1, max_tries + 1):
        try:
            response = requests.get(f"{API}/{kind}/search", params=params, headers=HEADERS, timeout=25)
            if response.status_code == 200:
                return response.json().get("data", [])
            wait = attempt * 4 if response.status_code in [429, 422] else 3
            print(f"    Server answered {response.status_code}, waiting {wait} s")
            time.sleep(wait)
        except Exception as error:
            print(f"    Network error: {error}, trying again in 4 s")
            time.sleep(4)
    return []


def make_record(item, record_id, thread_id, event, subreddit, kind, text, upvote_ratio):
    return {"record_id": record_id, "parent_post_id": thread_id, "event": event,
            "subreddit": subreddit, "type": kind, "created_utc": int(item.get("created_utc", 0)),
            "author_hash": anonymize_user(item.get("author")), "text": text,
            "score": int(item.get("score", 0)), "upvote_ratio": upvote_ratio}


def find_posts(subreddit, event, target):
    """Stage 1: posts of the time window whose title or text contains a keyword."""
    window = EVENTS[event]
    # Filtering by keywords here (not on the server) avoids server timeouts
    pattern = re.compile(f"({window['q']}|Ukraine|Russia|Ukrainian|Russian)", re.IGNORECASE)
    # Four points inside the window, so that one day does not dominate
    step = (window["before"] - window["after"]) // 4
    records, thread_ids = [], []
    for number in range(4):
        if len(thread_ids) >= max(25, target // 35):      # enough threads
            break
        params = {"subreddit": subreddit, "after": window["after"],
                  "before": window["before"] - number * step, "limit": 100, "sort": "desc"}
        posts = api_get("posts", params)
        time.sleep(1.2)      # polite pause between requests
        for post in posts:
            text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()
            post_id = post.get("id")
            if pattern.search(text) and post_id and post_id not in thread_ids:
                thread_ids.append(post_id)
                ratio = float(post.get("upvote_ratio") or 0.0)
                records.append(make_record(post, f"t3_{post_id}", str(post_id), event,
                                           subreddit, "post", text, ratio))
    return records, thread_ids


def scrape(subreddit, event, target):
    """Stage 2: add the comments of the found threads until the target is reached."""
    records, thread_ids = find_posts(subreddit, event, target)
    for thread_id in thread_ids:
        if len(records) >= target:
            break
        params = {"link_id": thread_id, "limit": min(100, target - len(records)), "sort": "desc"}
        comments = api_get("comments", params)
        time.sleep(1.2)
        for comment in comments[:target - len(records)]:
            records.append(make_record(comment, f"t1_{comment.get('id')}",
                                       str(comment.get("link_id", "")).replace("t3_", ""), event,
                                       subreddit, "comment", str(comment.get("body", "")).strip(), 0.0))
    return pd.DataFrame(records[:target])


def main():
    if RAW_FILE.exists() or CLEAN_FILE.exists():
        print("The data already exists, nothing is downloaded.")
        return
    parts = []
    for event in EVENTS:
        for subreddit, target in quotas_for_event(event).items():
            if target == 0:
                continue
            print(f"Downloading {event} | r/{subreddit} (target {target})")
            part = scrape(subreddit, event, target)
            print(f"  got {len(part)} records")
            if not part.empty:
                parts.append(part)
        if parts:      # save after every event
            RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(parts, ignore_index=True).to_csv(RAW_FILE, index=False)
    print(f"Done: {sum(len(part) for part in parts)} raw records")


if __name__ == "__main__":
    main()
