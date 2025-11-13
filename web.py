import pandas as pd

# Sample web-scraped data
web_data = [
    {"title": "You Play Rocket League in R6", "views": 35611, "likes": 623, "comments": 7, "video_length": 10},
    {"title": "R6 vs Rocket League Pros", "views": 28941, "likes": 1356, "comments": 57, "video_length": 15},
    {"title": "Rainbow Six Siege Champ Hits Champ in Rocket League", "views": 11811, "likes": 164, "comments": 42, "video_length": 20},
    {"title": "The Crossover We Needed #rainbowsixsiege #r6", "views": 34813, "likes": 738, "comments": 9, "video_length": 12},
    {"title": "Rocket League Madness!", "views": 50000, "likes": 1024, "comments": 50, "video_length": 8}
]

df_web = pd.DataFrame(web_data)

# Save CSV
csv_path = "/Users/jto5108/cmpsc455-project1/Youtube_Video_Data.csv"
df_web.to_csv("Youtube_Video_Data.csv", index=False)
print("Web CSV created: Youtube_Video_Data.csv")
