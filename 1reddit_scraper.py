import os
import time
import re
import hashlib
import requests
import pandas as pd

# Configuration dictionary defining the five target war events.
# Each entry maps an event identifier to a 28-day Unix UTC window (one week prior
# to three weeks post-event) and a pipe-delimited string of event-specific keywords.
EVENTS = {
    "1_invasion": {"after": 1645056000, "before": 1647561600, "q": "Ukraine|Russia|invasion|Kyiv"},
    "2_bucha": {"after": 1648166400, "before": 1650672000, "q": "Bucha|massacre|Irpin|war crimes"},
    "3_nordstream": {"after": 1663545600, "before": 1666051200, "q": "Nord Stream|pipeline|sabotage"},
    "4_grain": {"after": 1694131200, "before": 1696636800, "q": "Poland|grain|farmers|embargo"},
    "5_carlson_aid": {"after": 1706745600, "before": 1709251200, "q": "Tucker|Carlson|aid|funding|Putin"}
}

# Target raw record counts per subreddit for each event.
# Quotas include a ~25% buffer above the 6,000-record target to offset downstream
# attrition from deleted comments, bots, and short-text filtering.
BASE_SUBREDDIT_QUOTAS = {
    "worldnews": 2500,
    "europe": 1900,
    "UkraineRussiaReport": 1900,
    "Conservative": 650,
    "politics": 650
}

def anonymize_user(author_obj):
    """
    Converts a Reddit username into a deterministic 12-character SHA-256 hash.
    Preserves author-level grouping for spam detection while stripping personally
    identifiable handles. Deleted accounts and bots are mapped to a shared label.
    """
    if not author_obj or str(author_obj) in ["[deleted]", "AutoModerator", "None"]:
        return "deleted_or_bot"
    return hashlib.sha256(str(author_obj).encode("utf-8")).hexdigest()[:12]

def get_quotas_for_event(event_key, test_mode=False):
    """
    Calculates subreddit sample quotas for a given event.
    Because r/UkraineRussiaReport was inactive during the initial February 2022 invasion,
    its quota is zeroed out for '1_invasion' and split evenly between r/worldnews and
    r/europe to maintain a uniform total sample size across all five events.
    """
    if test_mode:
        # Small 50-record cap per active subreddit used during pipeline verification runs
        quotas = {sub: 50 for sub in BASE_SUBREDDIT_QUOTAS}
        if event_key == "1_invasion":
            quotas["UkraineRussiaReport"] = 0
        return quotas

    quotas = BASE_SUBREDDIT_QUOTAS.copy()
    if event_key == "1_invasion":
        urr_share = quotas["UkraineRussiaReport"]
        quotas["UkraineRussiaReport"] = 0
        # Even redistribution of the inactive subreddit's share across the two main news subs
        quotas["worldnews"] += urr_share // 2
        quotas["europe"] += urr_share - (urr_share // 2)

    return quotas

def safe_api_get(url, params, headers, max_retries=5):
    """
    Executes an HTTP GET request against the Arctic Shift API with retry logic.
    Applies linear backoff delays when encountering HTTP 429 (rate limit) or
    HTTP 422 (database timeout) responses rather than dropping the batch.
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
    Collects posts and comments for a subreddit-event pair via a two-stage strategy:
      1. Queries posts purely by indexed timestamps across four weekly checkpoints
         and filters titles/bodies locally using compiled regular expressions.
      2. Retrieves comments belonging to those matched posts via indexed 'link_id' lookups.
    """
    headers = {"User-Agent": "UkraineOpinionStudy/1.0 (Academic Research Project)"}

    # Compiled regex patterns used for local in-memory filtering of post text
    keyword_pattern = re.compile(f"({query}|Ukraine|Russia|Ukrainian|Russian)", re.IGNORECASE)
    strict_pattern = re.compile(f"({query})", re.IGNORECASE)

    records = []
    matched_post_ids = []

    posts_url = "https://arctic-shift.photon-reddit.com/api/posts/search"

    # Four evenly spaced weekly upper bounds across the 28-day window to prevent single-day clustering
    step = (before_ts - after_ts) // 4
    checkpoints = [before_ts, before_ts - step, before_ts - 2 * step, before_ts - 3 * step]

    # Stage 1: Timestamp-based post discovery and local keyword matching
    for cp_before in checkpoints:
        # Target thread count scales dynamically with the desired sample quota
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
        time.sleep(1.2)  # Throttles request rate to avoid triggering server-side timeouts

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

    # Stage 2: Indexed comment retrieval for each matched submission ID
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

if __name__ == "__main__":
    # Output directory for raw intermediate CSV checkpoints
    os.makedirs("data/raw", exist_ok=True)

    # Flag controlling whether the script runs a 50-row test pass or the full production scrape
    TEST_MODE = False

    all_dataframes = []

    for event_key, event_info in EVENTS.items():
        # Resolves active subreddit quotas for the current event window
        event_quotas = get_quotas_for_event(event_key, test_mode=TEST_MODE)

        for sub, target in event_quotas.items():
            if target == 0:
                print(f"Skipping: {event_key} | r/{sub} (Subreddit did not exist in Feb 2022 - quota redistributed)")
                continue

            print(f"Scraping: {event_key} | r/{sub} (Target: {target} records)...")

            df_part = scrape_without_422(
                subreddit=sub,
                event_key=event_key,
                after_ts=event_info["after"],
                before_ts=event_info["before"],
                query=event_info["q"],
                target_size=target
            )
            print(f"  -> Collected: {len(df_part)} records")

            if not df_part.empty:
                all_dataframes.append(df_part)

        # Incremental checkpoint save after each event completes
        if all_dataframes:
            df_raw_all = pd.concat(all_dataframes, ignore_index=True)
            df_raw_all.to_csv("data/raw/reddit_raw_checkpoint.csv", index=False)

    print(f"\nDONE! Total raw records collected: {len(df_raw_all)}")
