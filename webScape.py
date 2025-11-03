from googleapiclient.discovery import build
import pandas as pd
from dotenv import load_dotenv
import os

def fetch_youtube_data(api_key, query="technology", max_results=50):
    youtube = build("youtube", "v3", developerKey=api_key)

    request = youtube.search().list(
        q=query, part="snippet", type="video", maxResults=max_results
    )
    response = request.execute()

    videos = []
    for item in response["items"]:
        video_id = item["id"]["videoId"]
        snippet = item["snippet"]

        stats_request = youtube.videos().list(
            part="statistics,snippet", id=video_id
        )
        stats_response = stats_request.execute()

        if not stats_response["items"]:
            continue
        vid = stats_response["items"][0]
        stat = vid.get("statistics", {})

        videos.append({
            "title": snippet.get("title"),
            "channel": snippet.get("channelTitle"),
            "views": stat.get("viewCount", 0),
            "likes": stat.get("likeCount", 0),
            "comments": stat.get("commentCount", 0),
            "upload_date": snippet.get("publishedAt"),
            "category": snippet.get("categoryId", "Unknown"),
        })

    df = pd.DataFrame(videos)
    df.to_csv("data/youtube_api_data.csv", index=False)
    print(f"Saved {len(df)} API videos to data/youtube_api_data.csv")
    return df

if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("YOUTUBE_API_KEY")
    fetch_youtube_data(key)

