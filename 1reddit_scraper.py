import os
import time
import re
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
    if not author_obj or str(author_obj) in ["[deleted]", "AutoModerator", "None"]:
        return "deleted_or_bot"
    return hashlib.sha256(str(author_obj).encode("utf-8")).hexdigest()[:12]

def safe_api_get(url, params, headers, max_retries=4):
    """
    Executes an HTTP GET request against the Arctic Shift API with retry logic
    for HTTP 429 rate-limit responses.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=25)
            if resp.status_code == 200:
                return resp.json().get("data", [])
            elif resp.status_code == 429:
                wait = attempt * 5
                print(f"    [429 Rate Limit] Waiting {wait}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)
            elif resp.status_code == 422:
                print(f"    [422 Detail] {resp.text[:120]}")
                return []
            else:
                time.sleep(3)
        except Exception as e:
            print(f"    [Network Error] {e}, retrying in 4s...")
            time.sleep(4)
    return []

def scrape_without_422(subreddit, event_key, after_ts, before_ts, query, target_size):
    """
    Collects posts purely by indexed timestamps across four weekly checkpoints,
    filters titles locally via regex, and pulls comments via indexed 'link_id'.
    """
    headers = {"User-Agent": "UkraineOpinionStudy/1.0 (Academic Research Project)"}
    keyword_pattern = re.compile(f"({query}|Ukraine|Russia|Ukrainian|Russian)", re.IGNORECASE)
    strict_pattern = re.compile(f"({query})", re.IGNORECASE)

    records = []
    matched_post_ids = []
    posts_url = "https://arctic-shift.photon-reddit.com/api/posts/search"

    step = (before_ts - after_ts) // 4
    checkpoints = [before_ts, before_ts - step, before_ts - 2 * step, before_ts - 3 * step]

    for cp_before in checkpoints:
        if len(matched_post_ids) >= 25:
            break
        posts_params = {
            "subreddit": subreddit,
            "after": after_ts,
            "before": cp_before,
            "limit": 100,
            "sort": "desc"
        }
        batch_posts = safe_api_get(posts_url, posts_params, headers)
        time.sleep(0.8)

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
        time.sleep(0.8)

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

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    TEST_MODE = True
    all_dataframes = []

    for event_key, event_info in EVENTS.items():
        for sub, full_quota in SUBREDDIT_QUOTAS.items():
            target = 50 if TEST_MODE else full_quota
            df_part = scrape_without_422(
                subreddit=sub,
                event_key=event_key,
                after_ts=event_info["after"],
                before_ts=event_info["before"],
                query=event_info["q"],
                target_size=target
            )
            if not df_part.empty:
                all_dataframes.append(df_part)

        if all_dataframes:
            df_raw_all = pd.concat(all_dataframes, ignore_index=True)
            df_raw_all.to_csv("data/raw/reddit_raw_checkpoint.csv", index=False)
