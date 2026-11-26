import os
import time
import hashlib
import requests
import pandas as pd

# Configuration dictionary defining the five target war events and 28-day Unix UTC windows.
EVENTS = {
    "1_invasion": {"after": 1645056000, "before": 1647561600, "q": "Ukraine|Russia|invasion|Kyiv"},
    "2_bucha": {"after": 1648166400, "before": 1650672000, "q": "Bucha|massacre|Irpin|war crimes"},
    "3_nordstream": {"after": 1663545600, "before": 1666051200, "q": "Nord Stream|pipeline|sabotage"},
    "4_grain": {"after": 1694131200, "before": 1696636800, "q": "Poland|grain|farmers|embargo"},
    "5_carlson_aid": {"after": 1706745600, "before": 1709251200, "q": "Tucker|Carlson|aid|funding|Putin"}
}

# Target record quotas per subreddit for each event.
SUBREDDIT_QUOTAS = {
    "worldnews": 2000,
    "europe": 1500,
    "UkraineRussiaReport": 1500,
    "Conservative": 500,
    "politics": 500
}

def anonymize_user(author_obj):
    """
    Converts a Reddit username into a deterministic 12-character SHA-256 hash.
    """
    if not author_obj or str(author_obj) in ["[deleted]", "AutoModerator"]:
        return "deleted_or_bot"
    return hashlib.sha256(str(author_obj).encode("utf-8")).hexdigest()[:12]

def scrape_event_subreddit_paginated(subreddit, event_key, after_ts, before_ts, query, target_size):
    """
    Queries the Arctic Shift API in 100-record batches using server-side keyword
    parameters and backward timestamp pagination ('before').
    """
    headers = {"User-Agent": "UkraineOpinionStudy/1.0 (University Text Analytics Project)"}
    primary_keyword = query.split("|")[0].strip()
    records = []

    posts_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    posts_params = {
        "subreddit": subreddit,
        "after": after_ts,
        "before": before_ts,
        "title": primary_keyword,
        "limit": min(100, max(10, target_size // 10)),
        "sort": "desc"
    }
    try:
        resp_p = requests.get(posts_url, params=posts_params, headers=headers, timeout=20)
        if resp_p.status_code == 200:
            for p in resp_p.json().get("data", []):
                records.append({
                    "record_id": f"t3_{p.get('id')}",
                    "parent_post_id": p.get("id"),
                    "event": event_key,
                    "subreddit": subreddit,
                    "type": "post",
                    "created_utc": int(p.get("created_utc", 0)),
                    "author_hash": anonymize_user(p.get("author")),
                    "text": f"{p.get('title', '')} {p.get('selftext', '')}".strip(),
                    "score": p.get("score", 0),
                    "upvote_ratio": p.get("upvote_ratio", None)
                })
    except Exception as e:
        print(f"Error fetching posts ({subreddit}): {e}")

    comments_url = "https://arctic-shift.photon-reddit.com/api/comments/search"
    current_before = before_ts

    while len(records) < target_size:
        batch_limit = min(100, target_size - len(records))
        comments_params = {
            "subreddit": subreddit,
            "after": after_ts,
            "before": current_before,
            "body": primary_keyword,
            "limit": batch_limit,
            "sort": "desc"
        }
        try:
            resp_c = requests.get(comments_url, params=comments_params, headers=headers, timeout=20)
            if resp_c.status_code != 200:
                print(f"Server returned status {resp_c.status_code}, waiting 3 seconds...")
                time.sleep(3)
                break

            data = resp_c.json().get("data", [])
            if not data:
                break

            for c in data:
                records.append({
                    "record_id": f"t1_{c.get('id')}",
                    "parent_post_id": str(c.get("link_id", "")).replace("t3_", ""),
                    "event": event_key,
                    "subreddit": subreddit,
                    "type": "comment",
                    "created_utc": int(c.get("created_utc", 0)),
                    "author_hash": anonymize_user(c.get("author")),
                    "text": str(c.get("body", "")).strip(),
                    "score": c.get("score", 0),
                    "upvote_ratio": None
                })

            oldest_timestamp = int(data[-1].get("created_utc", current_before))
            current_before = (current_before - 1) if oldest_timestamp >= current_before else (oldest_timestamp - 1)
            time.sleep(0.5)

        except Exception as e:
            print(f"Error fetching comments ({subreddit}): {e}")
            break

    return pd.DataFrame(records)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    TEST_MODE = True
    all_dataframes = []

    for event_key, event_info in EVENTS.items():
        for sub, full_quota in SUBREDDIT_QUOTAS.items():
            target = 50 if TEST_MODE else full_quota
            df_part = scrape_event_subreddit_paginated(
                subreddit=sub,
                event_key=event_key,
                after_ts=event_info["after"],
                before_ts=event_info["before"],
                query=event_info["q"],
                target_size=target
            )
            all_dataframes.append(df_part)

        df_raw_all = pd.concat(all_dataframes, ignore_index=True)
        df_raw_all.to_csv("data/raw/reddit_raw_checkpoint.csv", index=False)
