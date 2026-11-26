import praw
import pandas as pd

# Initializes the official Reddit API wrapper using local credentials.
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_SECRET",
    user_agent="text_analytics_project_v1.0"
)

def fetch_event_data(subreddit_name, search_query, limit=500):
    """
    Searches a given subreddit for top posts matching the search query
    and collects post-level metadata into a pandas DataFrame.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []

    for post in subreddit.search(search_query, sort="top", limit=limit):
        posts_data.append({
            "id": post.id,
            "title": post.title,
            "score": post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "created_utc": post.created_utc
        })
    return pd.DataFrame(posts_data)

if __name__ == "__main__":
    # Initial test pull for the Bucha event in r/europe
    bucha_df = fetch_event_data("europe", "Bucha", limit=500)
