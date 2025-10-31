# webScrape.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def scrape_youtube_trending():
    """Scrape YouTube trending page for video data"""
    url = "https://www.youtube.com/feed/trending"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    videos = []
    for video in soup.find_all("a", id="video-title"):
        title = video.get("title")
        link = "https://www.youtube.com" + video.get("href")
        videos.append({"title": title, "url": link})

    print(f"✅ Collected {len(videos)} trending videos.")
    return pd.DataFrame(videos)

def save_scraped_data(df, filename="youtube_trending_data.csv"):
    """Save scraped data to CSV"""
    os.makedirs("data", exist_ok=True)
    output_file = os.path.join("data", filename)
    df.to_csv(output_file, index=False)
    print(f"💾 Data saved to {output_file}")
