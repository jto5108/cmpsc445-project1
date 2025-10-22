# cmpsc445-project1
Develop two machine learning models to predict video category and identify key factors influencing engagement. One model will use data collected via web scraping, and the other will use data from the YouTube Data API.


Required Works

1. Data Collection
  -Use web scraping tools to collect job postings from the Youtube websites.
  -Use the YouTube Data API with an API key to collect structured data
  -Ensure overlap or similarity in the video sets for fair comparison
  -Collect at least 3000 data for each
2. Data Preprocessing
  -Clean and normalize both datasets.
  -Handle missing or inconsistent values.
  -Standardize features (e.g., convert duration to minutes, normalize view counts).
3. Feature Engineering
  -Create features like video length, time since upload, keyword frequency in title/description, channel subscriber count (API only), or engagement rate = (likes + comments) / views 
4. Model Development
  -Train two regression models (e.g., Random Forest, XGBoost) to predict view count or engagement rate.
  -Compare performance between models trained on scraped vs. API data.
  -Use feature importance techniques to identify the most important attributes  (You can simply use the function supported by SKLearn library)
5. Visualization
  -Compare prediction accuracy between the two models.
  -Visualize feature importance for each model.
  -Show engagement trends by video category, length, or upload time.
