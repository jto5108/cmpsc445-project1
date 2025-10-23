
# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Load Data
# -------------------------------
web_df = pd.read_csv("youtube_trending_data.csv")
api_df = pd.read_csv("youtube_api_data.csv")

# -------------------------------
# Preprocess (example simplification)
# -------------------------------
web_df["title_length"] = web_df["title"].apply(lambda x: len(str(x)))
api_df["title_length"] = api_df["title"].apply(lambda x: len(str(x)))

# Dummy numeric target (since real views are missing in web scrape)
web_df["views"] = np.random.randint(1000, 1000000, len(web_df))
api_df["views"] = np.random.randint(1000, 1000000, len(api_df))

# -------------------------------
# Train Model Function
# -------------------------------
def train_model(df, label):
    X = df[["title_length"]]
    y = df["views"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ {label} Model R²: {r2:.3f}")

    plt.bar(["title_length"], model.feature_importances_)
    plt.title(f"Feature Importance ({label})")
    plt.show()

# -------------------------------
# Train Both Models
# -------------------------------
train_model(web_df, "Web-Scraped Data")
train_model(api_df, "YouTube API Data")
