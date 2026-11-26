import os
import time
import re
import hashlib
import requests
import pandas as pd

# Define the 5 war events we are studying.
# Timestamps are in Unix UTC format (1 week before to 3 weeks after each event).
# The 'q' string acts as an OR filter so we only grab threads relevant to that specific event.
EVENTS = {
    "1_invasion": {"after": 1645056000, "before": 1647561600, "q": "Ukraine|Russia|invasion|Kyiv"},
    "2_bucha": {"after": 1648166400, "before": 1650672000, "q": "Bucha|massacre|Irpin|war crimes"},
    "3_nordstream": {"after": 1663545600, "before": 1666051200, "q": "Nord Stream|pipeline|sabotage"},
    "4_grain": {"after": 1694131200, "before": 1696636800, "q": "Poland|grain|farmers|embargo"},
    "5_carlson_aid": {"after": 1706745600, "before": 1709251200, "q": "Tucker|Carlson|aid|funding|Putin"}
}

def anonymize_user(author_obj):
    """
    Turn usernames into irreversible 12-character hashes for GDPR/rubric privacy compliance.
    We still get a consistent ID per user to check if one spammer is dominating a thread,
    without storing anyone's real Reddit handle.
    """
    if not author_obj or str(author_obj) in ["[deleted]", "AutoModerator", "None"]:
        return "deleted_or_bot"
    return hashlib.sha256(str(author_obj).encode("utf-8")).hexdigest()[:12]


import os
import time
import re
import requests
import pandas as pd

# Base quotas per event (increased by ~25% so you still have 6,000+ clean rows after removing bots/deleted comments)
BASE_SUBREDDIT_QUOTAS = {
    "worldnews": 2500,
    "europe": 1900,
    "UkraineRussiaReport": 1900,
    "Conservative": 650,
    "politics": 650
}

def get_quotas_for_event(event_key, test_mode=True):
    """
    Returns the target sample size for each subreddit for a given event.
    Because r/UkraineRussiaReport did not exist during the initial invasion (Feb 2022),
    we skip it for '1_invasion' and split its quota evenly between r/worldnews and r/europe
    so Event 1 still hits the exact same total sample size as the other events.
    """
    if test_mode:
        # In test mode, pull 50 rows per active subreddit (and 0 for URR during Event 1)
        quotas = {sub: 50 for sub in BASE_SUBREDDIT_QUOTAS}
        if event_key == "1_invasion":
            quotas["UkraineRussiaReport"] = 0
        return quotas

    # Full production quotas
    quotas = BASE_SUBREDDIT_QUOTAS.copy()
    if event_key == "1_invasion":
        urr_share = quotas["UkraineRussiaReport"]
        quotas["UkraineRussiaReport"] = 0
        # Redistribute URR's quota equally to worldnews and europe to keep Event 1 balanced
        quotas["worldnews"] += urr_share // 2
        quotas["europe"] += urr_share - (urr_share // 2)

    return quotas

def safe_api_get(url, params, headers, max_retries=5):
    """
    Sends a GET request to the Arctic Shift API.
    Handles both 429 (Too Many Requests) and 422 ("Timeout. Maybe slow down a bit")
    by backing off and retrying instead of dropping the batch.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=25)
            if resp.status_code == 200:
                return resp.json().get("data", [])
            elif resp.status_code in [429, 422]:
                wait = attempt * 4
                print(f"    [Status {resp.status_code} - Rate Limit/Timeout] Backing off for {wait}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)
            else:
                print(f"    [Status {resp.status_code}] Unexpected response, waiting 3s...")
                time.sleep(3)
        except Exception as e:
            print(f"    [Network Error] {e}, retrying in 4s...")
            time.sleep(4)
    return []

def scrape_without_422(subreddit, event_key, after_ts, before_ts, query, target_size):
    """
    Two-stage scraper:
    1. Fetches post batches purely by timestamp (indexed on the server, avoiding 422 full-text timeouts)
       and filters for event keywords locally in Python using regex.
    2. Pulls comments directly from those matched threads via 'link_id'.
    """
    headers = {"User-Agent": "UkraineOpinionStudy/1.0 (Academic Research Project)"}

    # Compile regex patterns to filter post titles/bodies locally in memory
    keyword_pattern = re.compile(f"({query}|Ukraine|Russia|Ukrainian|Russian)", re.IGNORECASE)
    strict_pattern = re.compile(f"({query})", re.IGNORECASE)

    records = []
    matched_post_ids = []

    posts_url = "https://arctic-shift.photon-reddit.com/api/posts/search"

    # Sample across 4 weekly checkpoints inside the 28-day window so one single day doesn't dominate
    step = (before_ts - after_ts) // 4
    checkpoints = [before_ts, before_ts - step, before_ts - 2 * step, before_ts - 3 * step]

    # STAGE 1: Find relevant thread IDs in this time window
    for cp_before in checkpoints:
        # Scale how many threads we need based on target_size (roughly ~40-60 comments per thread)
        max_threads_needed = max(25, target_size // 35)
        if len(matched_post_ids) >= max_threads_needed:
            break

        posts_params = {
            "subreddit": subreddit,
            "after": after_ts,
            "before": cp_before,
            "limit": 100,
            "sort": "desc"
        }

        batch_posts = safe_api_get(posts_url, posts_params, headers)
        time.sleep(1.2) # Polite delay between requests

        for p in batch_posts:
            full_title_text = f"{p.get('title', '')} {p.get('selftext', '')}".strip()

            if strict_pattern.search(full_title_text) or keyword_pattern.search(full_title_text):
                pid = p.get("id")
                if pid and pid not in matched_post_ids:
                    matched_post_ids.append(pid)
                    records.append({
                        "record_id": f"t3_{pid}",
                        "parent_post_id": str(pid),
                        "event": event_key,
                        "subreddit": subreddit,
                        "type": "post",
                        "created_utc": int(p.get("created_utc", 0)),
                        "author_hash": anonymize_user(p.get("author")),
                        "text": full_title_text,
                        "score": int(p.get("score", 0)),
                        "upvote_ratio": float(p.get("upvote_ratio", 0.0)) if p.get("upvote_ratio") is not None else 0.0
                    })

    # STAGE 2: Pull comments from each matched thread via indexed 'link_id'
    comments_url = "https://arctic-shift.photon-reddit.com/api/comments/search"

    for pid in matched_post_ids:
        if len(records) >= target_size:
            break

        com_params = {
            "link_id": pid,
            "limit": min(100, target_size - len(records)),
            "sort": "desc"
        }
        coms_data = safe_api_get(comments_url, com_params, headers)
        time.sleep(1.2)

        for c in coms_data:
            if len(records) >= target_size:
                break
            records.append({
                "record_id": f"t1_{c.get('id')}",
                "parent_post_id": str(c.get("link_id", "")).replace("t3_", ""),
                "event": event_key,
                "subreddit": subreddit,
                "type": "comment",
                "created_utc": int(c.get("created_utc", 0)),
                "author_hash": anonymize_user(c.get("author")),
                "text": str(c.get("body", "")).strip(),
                "score": int(c.get("score", 0)),
                "upvote_ratio": 0.0
            })

    return pd.DataFrame(records[:target_size])

# Ensure output folder exists
os.makedirs("data/raw", exist_ok=True)

# Set TEST_MODE = False when you are ready to run the full ~30,000-row production scrape
TEST_MODE = False

all_dataframes = []

for event_key, event_info in EVENTS.items():
    # Get dynamic quotas for this event (skips r/UkraineRussiaReport during 1_invasion)
    event_quotas = get_quotas_for_event(event_key, test_mode=TEST_MODE)

    for sub, target in event_quotas.items():
        if target == 0:
            print(f"Přeskakuji: {event_key} | r/{sub} (Subreddit v únoru 2022 ještě neexistoval – kvóta přesunuta)")
            continue

        print(f"Stahuji: {event_key} | r/{sub} (Cíl: {target} záznamů)...")

        df_part = scrape_without_422(
            subreddit=sub,
            event_key=event_key,
            after_ts=event_info["after"],
            before_ts=event_info["before"],
            query=event_info["q"],
            target_size=target
        )
        print(f"  -> Staženo: {len(df_part)} záznamů")

        if not df_part.empty:
            all_dataframes.append(df_part)

    # Save checkpoint after each event finishes
    if all_dataframes:
        df_raw_all = pd.concat(all_dataframes, ignore_index=True)
        df_raw_all.to_csv("data/raw/reddit_raw_checkpoint.csv", index=False)

print(f"\nHOTOVO! Celkem staženo {len(df_raw_all)} hrubých záznamů.")
