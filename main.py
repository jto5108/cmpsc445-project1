import os, time, pandas as pd
from dotenv import load_dotenv
from webScrape import scrape_youtube_from_google
from youtubeScrape import fetch_youtube_data
from train_models import train_model

def main():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")

    print("=== DATA COLLECTION ===")
    start = time.time()
    df_web = scrape_youtube_from_google(query="technology videos", max_results=10)
    df_api = fetch_youtube_data(api_key, query="technology", max_results=20)
    print(f"Data collection finished in {time.time()-start:.1f}s\n")

    print("=== MODEL TRAINING ===")
    model_web, _, _ = train_model(df_web, "Web-Scraped Data")
    model_api, _, _ = train_model(df_api, "YouTube API Data")

    print("=== COMPARISON SUMMARY ===")
    summary = pd.DataFrame([
        {"Source": "Web-Scraped", "Samples": len(df_web)},
        {"Source": "API", "Samples": len(df_api)}
    ])
    print(summary)
    summary.to_csv("data/comparison_summary.csv", index=False)

if __name__ == "__main__":
    main()
