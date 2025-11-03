import os
import sys
import time
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from googleapiclient.discovery import build
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ----------------------------
# Web Scraping Functions
# ----------------------------
def get_youtube_links_from_google(query="technology videos", max_results=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}+site:youtube.com&num={max_results}"
    soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")
    links = []
    for a in soup.select("a[href^='https://www.youtube.com/watch']"):
        href = a["href"].split("&")[0]
        if href not in links:
            links.append(href)
        if len(links) >= max_results:
            break
    return links

def scrape_youtube_video_page(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    s = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")
    
    def meta(prop):
        tag = s.find("meta", itemprop=prop)
        return tag["content"] if tag and tag.has_attr("content") else "N/A"
    
    return {
        "title": meta("name") or meta("title"),
        "channel": meta("channelId"),
        "views": "N/A",
        "likes": "N/A",
        "comments": "N/A",
        "upload_date": meta("datePublished"),
        "category": meta("genre"),
        "url": url
    }

def scrape_youtube_from_google(query="technology videos", max_results=5):
    links = get_youtube_links_from_google(query, max_results)
    data = []
    for i, link in enumerate(links, 1):
        try:
            print(f"Scraping {i}/{len(links)}: {link}")
            data.append(scrape_youtube_video_page(link))
            time.sleep(2)
        except Exception as e:
            print("Error:", e)
    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/youtube_scraped_data.csv", index=False)
    print(f"Saved {len(df)} scraped videos to data/youtube_scraped_data.csv")
    return df

# ----------------------------
# YouTube API Functions
# ----------------------------
def fetch_youtube_data(api_key, query="technology", max_results=5):
    youtube = build("youtube", "v3", developerKey=api_key)
    
    request = youtube.search().list(
        q=query, part="snippet", type="video", maxResults=max_results
    )
    response = request.execute()
    
    videos = []
    for item in response.get("items", []):
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
            "views": int(stat.get("viewCount", 0)),
            "likes": int(stat.get("likeCount", 0)),
            "comments": int(stat.get("commentCount", 0)),
            "upload_date": snippet.get("publishedAt"),
            "category": snippet.get("categoryId", "Unknown"),
            "url": f"https://www.youtube.com/watch?v={video_id}"
        })
    df = pd.DataFrame(videos)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/youtube_api_data.csv", index=False)
    print(f"Saved {len(df)} API videos to data/youtube_api_data.csv")
    return df

# ----------------------------
# Model Training Functions
# ----------------------------
def prepare_data(df):
    # Clean numeric fields
    for col in ["views", "likes", "comments"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace(r"\D", "", regex=True)
                .replace("", "0").astype(int)
            )
    # Text features
    tfidf = TfidfVectorizer(max_features=300)
    X_text = tfidf.fit_transform(df["title"].fillna("")).toarray()
    
    # Numeric features
    X_num = pd.DataFrame()
    X_num["views"] = df["views"].fillna(0)
    if "likes" in df.columns:
        X_num["likes"] = df["likes"].fillna(0)
    if "comments" in df.columns:
        X_num["comments"] = df["comments"].fillna(0)
    
    X = np.hstack([X_text, X_num.values])
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df["category"].fillna("Unknown"))
    
    return X, y, le, tfidf

def train_model(df, name=""):
    print(f"\nTraining model for: {name}")
    X, y, le, tfidf = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_[:5]))
    
    return model, le, tfidf

# ----------------------------
# Feature Importance Plot
# ----------------------------
def plot_feature_importance(model, df, tfidf, title="Feature Importance", top_n=10):
    numeric_features = ["views"]
    if "likes" in df.columns:
        numeric_features.append("likes")
    if "comments" in df.columns:
        numeric_features.append("comments")
    text_features = tfidf.get_feature_names_out()
    feature_names = list(text_features) + numeric_features
    
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(10,6))
    plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main Function
# ----------------------------
def main():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: YouTube API key missing in .env")
        return
    
    print("\n=== DATA COLLECTION ===")
    start = time.time()
    
    df_web = scrape_youtube_from_google(query="technology videos", max_results=5)
    print("\nWeb-Scraped Sample Data:")
    print(df_web.head(5))
    
    df_api = fetch_youtube_data(api_key, query="technology", max_results=5)
    print("\nYouTube API Sample Data:")
    print(df_api.head(5))
    
    print(f"\nData collection finished in {time.time() - start:.1f}s")
    
    print("\n=== MODEL TRAINING ===")
    model_web, le_web, tfidf_web = train_model(df_web, "Web-Scraped Data")
    model_api, le_api, tfidf_api = train_model(df_api, "YouTube API Data")
    
    print("\n=== FEATURE IMPORTANCE ===")
    plot_feature_importance(model_web, df_web, tfidf_web, "Web-Scraped Data Feature Importance")
    plot_feature_importance(model_api, df_api, tfidf_api, "YouTube API Data Feature Importance")
    
    print("\n=== COMPARISON SUMMARY ===")
    summary = pd.DataFrame([
        {"Source": "Web-Scraped", "Samples": len(df_web)},
        {"Source": "YouTube API", "Samples": len(df_api)}
    ])
    print(summary)
    os.makedirs("data", exist_ok=True)
    summary.to_csv("data/comparison_summary.csv", index=False)
    print("\nComparison summary saved to data/comparison_summary.csv")

if __name__ == "__main__":
    main()
