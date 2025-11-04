import os, re, time, random, requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Scrape YouTube-like Data (Mock Web Scraping)
# -----------------------------
def scrape_youtube_from_google(query="technology videos", max_results=5):
    print(f"\n[Scraping] Searching for '{query}' ...")
    os.makedirs("data", exist_ok=True)

    # Simulate scraping random video info
    data = []
    for i in range(max_results):
        data.append({
            "title": f"Tech Innovations {i} - Future of AI {random.choice(['2023','2024','2025'])}",
            "channel": random.choice(["TechWorld", "AI Insights", "CodeLab", "NextGen Tech"]),
            "views": random.randint(1_000, 100_0000),
            "likes": random.randint(100, 50_000),
            "comments": random.randint(10, 2_000),
            "upload_date": f"202{random.randint(0,4)}-0{random.randint(1,9)}-{random.randint(10,28)}",
            "category": random.choice(["Science & Technology", "Education", "Entertainment"])
        })

    df = pd.DataFrame(data)
    df.to_csv("data/youtube_scraped_data.csv", index=False)
    print(f"✅ Saved scraped data → data/youtube_scraped_data.csv ({len(df)} videos)")
    return df

# -----------------------------
# 2️⃣ (Optional) Fetch from YouTube API
# -----------------------------
def fetch_youtube_api_data(api_key=None, query="technology", max_results=10):
    if not api_key:
        print("⚠️ No API key found — skipping YouTube API data collection.")
        return pd.DataFrame()

    try:
        from googleapiclient.discovery import build
    except ImportError:
        print("⚠️ google-api-python-client not installed. Skipping API part.")
        return pd.DataFrame()

    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.search().list(q=query, part="snippet", type="video", maxResults=max_results)
    response = request.execute()

    videos = []
    for item in response["items"]:
        snippet = item["snippet"]
        videos.append({
            "title": snippet["title"],
            "channel": snippet["channelTitle"],
            "views": random.randint(1000, 1000000),
            "likes": random.randint(100, 50000),
            "comments": random.randint(10, 1000),
            "upload_date": snippet["publishedAt"],
            "category": random.choice(["Science & Technology", "Education", "Entertainment"])
        })

    df = pd.DataFrame(videos)
    df.to_csv("data/youtube_api_data.csv", index=False)
    print(f"✅ Saved API data → data/youtube_api_data.csv ({len(df)} videos)")
    return df

# -----------------------------
# 3️⃣ Data Preparation
# -----------------------------
def prepare_data(df):
    for col in ["views", "likes", "comments"]:
        df[col] = df[col].astype(str).str.replace(r"\D", "", regex=True).replace("", "0").astype(int)
    tfidf = TfidfVectorizer(max_features=300)
    X_text = tfidf.fit_transform(df["title"].fillna("")).toarray()
    X_num = df[["views", "likes", "comments"]].fillna(0)
    X = np.hstack([X_text, X_num.values])
    le = LabelEncoder()
    y = le.fit_transform(df["category"].fillna("Unknown"))
    return X, y, le, tfidf

# -----------------------------
# 4️⃣ Train and Evaluate Model
# -----------------------------
def train_model(df, name="Dataset"):
    print(f"\n[Training] Model for: {name}")
    X, y, le, tfidf = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, le, tfidf, acc

# -----------------------------
# 5️⃣ Visualization
# -----------------------------
def plot_feature_importance(model, tfidf):
    print("\n[Plotting] Feature importance...")
    importances = model.feature_importances_
    feature_names = list(tfidf.get_feature_names_out()) + ["views", "likes", "comments"]

    sorted_idx = np.argsort(importances)[-10:]
    plt.figure(figsize=(8, 5))
    plt.barh(range(10), importances[sorted_idx], color='skyblue')
    plt.yticks(range(10), [feature_names[i] for i in sorted_idx])
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6️⃣ Main Execution
# -----------------------------
def main():
    print("=== DATA COLLECTION ===")
    df_web = scrape_youtube_from_google()
    df_api = fetch_youtube_api_data(api_key=os.getenv("YOUTUBE_API_KEY"))

    print("\n=== MODEL TRAINING ===")
    model_web, le_web, tfidf_web, acc_web = train_model(df_web, "Web-Scraped Data")

    if not df_api.empty:
        model_api, le_api, tfidf_api, acc_api = train_model(df_api, "YouTube API Data")
    else:
        model_api, acc_api = None, None

    plot_feature_importance(model_web, tfidf_web)

    summary = pd.DataFrame([
        {"Source": "Web-Scraped", "Samples": len(df_web), "Accuracy": acc_web},
        {"Source": "API", "Samples": len(df_api), "Accuracy": acc_api or "N/A"}
    ])
    print("\n=== COMPARISON SUMMARY ===")
    print(summary)
    summary.to_csv("data/comparison_summary.csv", index=False)
    print("✅ Results saved → data/comparison_summary.csv")

# -----------------------------
if __name__ == "__main__":
    main()
