import pandas as pd

# Sample API data
api_data = [
    {"title": "You Play Rocket League in R6", "views": 35611, "likes": 623, "comments": 7, "video_length": 10, "tags": "gaming,r6,rocket league"},
    {"title": "R6 vs Rocket League Pros", "views": 28941, "likes": 1356, "comments": 57, "video_length": 15, "tags": "gaming,pros,rocket league"},
    {"title": "Rainbow Six Siege Champ Hits Champ in Rocket League", "views": 11811, "likes": 164, "comments": 42, "video_length": 20, "tags": "rainbow six,gaming,rocket league"},
    {"title": "The Crossover We Needed #rainbowsixsiege #r6", "views": 34813, "likes": 738, "comments": 9, "video_length": 12, "tags": "rainbow six,r6,gaming"},
    {"title": "Rocket League Madness!", "views": 50000, "likes": 1024, "comments": 50, "video_length": 8, "tags": "rocket league,gaming"}
]

df_api = pd.DataFrame(api_data)

# Save CSV
df_api.to_csv("youtube_video_data.csv", index=False)
print("API CSV created: youtube_video_data.csv")
