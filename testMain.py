import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import re

# ---------------------------
# CONFIGURATION
# ---------------------------
SEARCH_QUERY = "technology videos"
WEB_SCRAPE_LIMIT = 100
API_LIMIT = 100
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # Set your key in shell: export YOUTUBE_API_KEY="YOUR_KEY"

# ---------------------------
# WEB SCRAPING FUNCTION
# ---------------------------
def webscrape_youtube(query, max_results=100):
    print("[WebScrape] Collecting YouTube video data via Google search...")
    videos = []
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    start = 0

    while len(videos) < max_results:
        url = f"https://www.google.com/search?q={query}+site:youtube.com&start={start}"
        response = session.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        links = [a['href'].split('=')[1].split('&')[0] for a in soup.find_all('a') if 'youtube.com/watch' in a.get('href','')]
        if not links:
            break

        for vid_id in links:
            if len(videos) >= max_results:
                break
            if vid_id in [v['video_id'] for v in videos]:
                continue
            video_url = f"https://www.youtube.com/watch?v={vid_id}"
            try:
                vid_page = session.get(video_url, headers=headers)
                title_tag = BeautifulSoup(vid_page.text, "html.parser").find("meta", property="og:title")
                title = title_tag["content"] if title_tag else "Unknown"
                # Placeholder for more metrics if available
                videos.append({"video_id": vid_id, "title": title})
            except:
                continue
        start += 10

    print(f"[WebScrape] Collected {len(videos)} videos")
    return pd.DataFrame(videos)

# ---------------------------
# YOUTUBE API FUNCTION
# ---------------------------
def fetch_youtube_api(query, max_results=100):
    if not YOUTUBE_API_KEY:
        print("⚠️ Missing API key. Skipping API collection.")
        return pd.DataFrame()
    print("[YouTube API] Collecting structured video data...")
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    videos = []
    next_page = None

    while len(videos) < max_results:
        request = youtube.search().list(
            part="id,snippet",
            q=query,
            type="video",
            maxResults=min(50, max_results-len(videos)),
            pageToken=next_page
        )
        response = request.execute()

        for item in response["items"]:
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]

            stats_req = youtube.videos().list(part="statistics,snippet", id=video_id)
            stats_resp = stats_req.execute()
            if not stats_resp["items"]:
                continue
            stat = stats_resp["items"][0].get("statistics", {})
            videos.append({
                "video_id": video_id,
                "title": snippet.get("title"),
                "channel": snippet.get("channelTitle"),
                "views": int(stat.get("viewCount",0)),
                "likes": int(stat.get("likeCount",0)),
                "comments": int(stat.get("commentCount",0)),
                "publish_date": snippet.get("publishedAt")
            })
        next_page = response.get("nextPageToken")
        if not next_page:
            break

    print(f"[YouTube API] Collected {len(videos)} videos")
    return pd.DataFrame(videos)

# ---------------------------
# PREPROCESSING
# ---------------------------
def preprocess_data(df):
    df = df.copy()
    for col in ["views","likes","comments"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["engagement_rate"] = (df.get("likes",0) + df.get("comments",0)) / np.maximum(df.get("views",1),1)
    if "publish_date" in df.columns:
        df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
        df["days_since_upload"] = (datetime.now() - df["publish_date"]).dt.days.fillna(0)
    df["keyword_count"] = df["title"].str.count(r"\btechnology\b|\bgaming\b|\bvideo\b")
    return df

# ---------------------------
# MODEL TRAINING
# ---------------------------
def train_model(df, target="engagement_rate"):
    features = ["views","likes","comments","days_since_upload","keyword_count"]
    X = df[features].fillna(0).values
    y = df[target].fillna(0).values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)
    print(f"[Model] RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return model, features

# ---------------------------
# FEATURE IMPORTANCE PLOT
# ---------------------------
def plot_feature_importance(model, features, title="Feature Importance"):
    importance = model.feature_importances_
    plt.figure(figsize=(8,5))
    plt.barh(features, importance)
    plt.title(title)
    plt.show()

# ---------------------------
# MAIN
# ---------------------------
def main():
    start_time = time.time()

    # Web scrape
    df_web = webscrape_youtube(SEARCH_QUERY, WEB_SCRAPE_LIMIT)
    df_web = preprocess_data(df_web)
    print("\n[Training Web-Scrape Model]")
    model_web, features_web = train_model(df_web)
    plot_feature_importance(model_web, features_web, "Web-Scrape Feature Importance")

    # YouTube API
    df_api = fetch_youtube_api(SEARCH_QUERY, API_LIMIT)
    df_api = preprocess_data(df_api)
    if not df_api.empty:
        print("\n[Training API Model]")
        model_api, features_api = train_model(df_api)
        plot_feature_importance(model_api, features_api, "API Feature Importance")

    print(f"\n[Finished] Total time: {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    main()
