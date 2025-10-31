# youtubeScrape.py
from googleapiclient.discovery import build
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()  # load .env file

def fetch_youtube_data(query="music", max_results=50):
    """Fetch video metadata using YouTube Data API"""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("❌ No API key found! Please set YOUTUBE_API_KEY in your .env file.")

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
    os.makedirs("data", exist_ok=True)
    output_file = os.path.join("data", "youtube_api_data.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df)} videos from API to {output_file}")
    print("\n📊 Preview of fetched data:")
    print(df.head())
    return df

