
# youtubeScrape.py
from googleapiclient.discovery import build
import pandas as pd

def fetch_youtube_data(api_key, query="music", max_results=50):
    """Fetch video metadata using YouTube Data API"""
    youtube = build("youtube", "v3", developerKey=api_key)

    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    video_data = []
    for item in response["items"]:
        video = {
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "publish_date": item["snippet"]["publishedAt"],
            "description": item["snippet"]["description"]
        }
        video_data.append(video)

    df = pd.DataFrame(video_data)
    df.to_csv("youtube_api_data.csv", index=False)
    print(f"✅ Saved {len(df)} videos from API to youtube_api_data.csv")
    return df

if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY_HERE"
    fetch_youtube_data(API_KEY, query="technology")

