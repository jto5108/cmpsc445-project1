# webScrape.py
# webScrape.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import urllib.parse

def scrape_youtube_from_google(query="technology videos", max_results=20):
    """Scrape YouTube video links from Google search results"""
    search_query = f"site:youtube.com {query}"
    encoded_query = urllib.parse.quote(search_query)
    url = f"https://www.google.com/search?q={encoded_query}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/141.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    videos = []
    for link in soup.select("a"):
        href = link.get("href")
        if href and "youtube.com/watch" in href:
            # Clean up the URL
            clean_url = href.replace("/url?q=", "").split("&")[0]
            videos.append({"url": clean_url})

    # Remove duplicates and limit results
    videos = pd.DataFrame(videos).drop_duplicates().head(max_results)

    # Optional: extract video titles from URLs
    videos["title"] = videos["url"].apply(lambda u: urllib.parse.unquote(u.split("v=")[-1])[:50])

    print(f"✅ Found {len(videos)} YouTube video links from Google for '{query}'.")
    return videos

def save_scraped_data(df, filename="youtube_trending_data.csv"):
    """Save scraped data to CSV and show preview"""
    os.makedirs("data", exist_ok=True)
    output_file = os.path.join("data", filename)
    df.to_csv(output_file, index=False)
    print(f"💾 Data saved to {output_file}")

    # Show first 5 examples
    print("\n📊 Preview of first 5 scraped videos:")
    print(df.head())

if __name__ == "__main__":
    df = scrape_youtube_from_google(query="technology videos", max_results=20)
    save_scraped_data(df)
