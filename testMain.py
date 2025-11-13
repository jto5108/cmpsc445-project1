import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------
API_KEY = "AIzaSyDejAMtAIGAo4GLDNBar764UQ0Ty6Euago"
WEB_CSV = "Youtube_Video_Data.csv"
API_CSV = "youtube_video_data.csv"
TARGET = "engagement_rate"

# Define keywords for automatic extraction
KEYWORDS = ["technology", "gaming", "video", "xbox", "playstation", "nintendo",
            "minecraft", "vr", "rocket league", "rainbow six", "overwatch"]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def preprocess_data(df, is_api=False):
    # Remove emojis
    df["title"] = df["title"].astype(str).str.encode('ascii', errors='ignore').str.decode()
    
    # Fill missing numeric values
    df["views"] = pd.to_numeric(df["views"], errors='coerce').fillna(0)
    df["likes"] = pd.to_numeric(df["likes"], errors='coerce').fillna(0)
    df["comments"] = pd.to_numeric(df["comments"], errors='coerce').fillna(0)
    df["video_length"] = pd.to_numeric(df.get("video_length", 0), errors='coerce').fillna(0)

    # Feature: engagement rate
    df["engagement_rate"] = (df["likes"] + df["comments"]) / (df["views"] + 1)

    # Feature: keyword frequency
    if is_api and "tags" in df.columns:
        df["keyword_count"] = df.apply(
            lambda row: sum(row["title"].lower().count(k) + str(row["tags"]).lower().count(k) for k in KEYWORDS),
            axis=1
        )
    else:
        df["keyword_count"] = df["title"].str.lower().apply(
            lambda t: sum(t.count(k) for k in KEYWORDS)
        )

    df.fillna(0, inplace=True)
    return df

def train_model(df, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("Train R2:", r2_score(y_train, y_pred_train))
    print("Test R2:", r2_score(y_test, y_pred_test))
    
    # Feature importance plot
    importances = model.feature_importances_
    plt.figure(figsize=(8,5))
    plt.barh(features, importances)
    plt.title("Feature Importance")
    plt.show()
    
    return model

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading web-scraped data...")
    df_web = pd.read_csv(WEB_CSV)
    df_web = preprocess_data(df_web)

    print("Loading API data...")
    df_api = pd.read_csv(API_CSV)
    df_api = preprocess_data(df_api, is_api=True)

    features = ["views", "likes", "comments", "video_length", "keyword_count"]

    print("\nTraining model on Web-Scraped Data...")
    model_web = train_model(df_web, features, TARGET)

    print("\nTraining model on API Data...")
    model_api = train_model(df_api, features, TARGET)

    # Comparison summary
    summary = pd.DataFrame({
        "Source": ["Web-Scraped", "API"],
        "Samples": [len(df_web), len(df_api)],
        "R2_Train": [r2_score(df_web[TARGET], model_web.predict(df_web[features])),
                     r2_score(df_api[TARGET], model_api.predict(df_api[features]))]
    })
    print("\nComparison Summary:")
    print(summary)

if __name__ == "__main__":
    main()
