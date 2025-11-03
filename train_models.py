import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def prepare_data(df):
    # Clean up numeric fields
    for col in ["views", "likes", "comments"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"\D", "", regex=True)
                .replace("", "0")
                .astype(int)
            )

    # Text feature: video title
    tfidf = TfidfVectorizer(max_features=300)
    X_text = tfidf.fit_transform(df["title"].fillna("")).toarray()

    # Numeric features
    X_num = df[["views"]].fillna(0)
    if "likes" in df.columns:
        X_num["likes"] = df["likes"]
    if "comments" in df.columns:
        X_num["comments"] = df["comments"]

    # Combine text + numeric features
    import numpy as np
    X = np.hstack([X_text, X_num.values])

    # Encode target category
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

if __name__ == "__main__":
    # Load both datasets
    df_web = pd.read_csv("data/youtube_scraped_data.csv")
    df_api = pd.read_csv("data/youtube_api_data.csv")

    model_web, le_web, tfidf_web = train_model(df_web, "Web-Scraped Data")
    model_api, le_api, tfidf_api = train_model(df_api, "YouTube API Data")

    print("\nComparison complete.")
