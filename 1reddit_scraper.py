import os
import time
import hashlib
import requests
import praw
import pandas as pd

# Connects to the official Reddit API via environment variables.
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET", "YOUR_SECRET"),
    user_agent="UkraineOpinionStudy/1.0 (Academic Research Project)"
)

# Configuration dictionary defining the five target war events and 28-day Unix UTC windows.
EVENTS = {
    "1_invasion": {"after": 1645056000, "before": 1647561600, "q": "Ukraine|Russia|invasion|Kyiv"},
    "2_bucha": {"after": 1648166400, "before": 1650672000, "q": "Bucha|massacre|Irpin|war crimes"},
    "3_nordstream": {"after": 1663545600, "before": 1666051200, "q": "Nord Stream|pipeline|sabotage"},
    "4_grain": {"after": 1694131200, "before": 1696636800, "q": "Poland|grain|farmers|embargo"},
    "5_carlson_aid": {"after": 1706745600, "before": 1709251200, "q": "Tucker|Carlson|aid|funding|Putin"}
}

def anonymize_user(author_obj):
    """
    Converts a Reddit username into a deterministic 12-character SHA-256 hash
    for privacy compliance while preserving author-level grouping.
    """
    if not author_obj or str(author_obj) in ["[deleted]", "AutoModerator"]:
        return "deleted_or_bot"
    return hashlib.sha256(str(author_obj).encode("utf-8")).hexdigest()[:12]

def fetch_historical_post_ids(subreddit, after_ts, before_ts, query, max_posts=150):
    """
    Queries the PullPush archive API to retrieve historical submission IDs
    within the specified Unix timestamp boundaries.
    """
    url = "https://api.pullpush.io/reddit/search/submission/"
    post_ids = []

    while len(post_ids) < max_posts:
        params = {
            "subreddit": subreddit,
            "after": after_ts,
            "before": before_ts,
            "q": query,
            "sort_type": "score",
            "sort": "desc",
            "size": 100
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                print(f"PullPush warning ({resp.status_code}), pausing...")
                time.sleep(3)
                break
            data = resp.json().get("data", [])
            if not data:
                break
            for item in data:
                post_ids.append(item["id"])
            break
        except Exception as e:
            print(f"Archive fetch error: {e}")
            break

    return list(set(post_ids))

def collect_posts_and_comments(subreddit_name, event_key, post_ids, target_sample_size, comment_sort="top"):
    """
    Hydrates historical submission IDs via PRAW to extract post text
    and up to 40 top-level or nested comments per thread.
    """
    records = []
    for pid in post_ids:
        if len(records) >= target_sample_size:
            break
        try:
            submission = reddit.submission(id=pid)
            submission.comment_sort = comment_sort
            submission.comments.replace_more(limit=0)

            records.append({
                "record_id": f"t3_{submission.id}",
                "parent_post_id": submission.id,
                "event": event_key,
                "subreddit": subreddit_name,
                "type": "post",
                "sort_mode": comment_sort,
                "created_utc": int(submission.created_utc),
                "author_hash": anonymize_user(submission.author),
                "text": f"{submission.title} {submission.selftext}".strip(),
                "score": submission.score,
                "upvote_ratio": submission.upvote_ratio
            })

            for comment in submission.comments.list()[:40]:
                if len(records) >= target_sample_size:
                    break
                if not hasattr(comment, "body"):
                    continue
                records.append({
                    "record_id": f"t1_{comment.id}",
                    "parent_post_id": submission.id,
                    "event": event_key,
                    "subreddit": subreddit_name,
                    "type": "comment",
                    "sort_mode": comment_sort,
                    "created_utc": int(comment.created_utc),
                    "author_hash": anonymize_user(comment.author),
                    "text": comment.body.strip(),
                    "score": comment.score,
                    "upvote_ratio": None
                })
            time.sleep(0.6)
        except Exception as e:
            print(f"Failed on post {pid}: {e}")
            continue

    return pd.DataFrame(records)
