# main.py
import os
import pandas as pd
from youtubeScrape import fetch_youtube_data
from webScrape import scrape_youtube_trending, save_scraped_data

def main():
    os.makedirs("data", exist_ok=True)

    # -----------------------------
    # 1️⃣ YouTube API fetch
    # -----------------------------
    print("🔹 Starting YouTube API fetch...")
    api_df = fetch_youtube_data(query="technology")

    # -----------------------------
    # 2️⃣ Web scrape trending page
    # -----------------------------
    print("\n🔹 Starting YouTube Trending scrape...")
    trending_df = scrape_youtube_trending()
    save_scraped_data(trending_df)

    # -----------------------------
    # 3️⃣ Merge datasets
    # -----------------------------
    combined_file = os.path.join("data", "combined_youtube_data.csv")

    api_df["source"] = "API"
    trending_df["source"] = "Trending"

    combined_df = pd.merge(
        api_df, trending_df, how="outer", left_on="title", right_on="title", suffixes=("_api", "_trending")
    )

    combined_df.to_csv(combined_file, index=False)
    print(f"\n✅ Combined CSV saved to {combined_file}")
    print("\n📊 Preview of combined data:")
    print(combined_df.head())

    print("\n🎉 All tasks completed successfully!")

if __name__ == "__main__":
    main()

