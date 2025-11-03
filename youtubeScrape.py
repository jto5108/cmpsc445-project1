import requests, re, time, pandas as pd
from bs4 import BeautifulSoup

def get_youtube_links_from_google(query="technology videos", max_results=10):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}+site:youtube.com&num={max_results}"
    soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")
    links = []
    for a in soup.select("a[href^='https://www.youtube.com/watch']"):
        href = a["href"].split("&")[0]
        if href not in links:
            links.append(href)
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
    }

def scrape_youtube_from_google(query="technology videos", max_results=10):
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
    df.to_csv("data/youtube_scraped_data.csv", index=False)
    print(f"Saved {len(df)} scraped videos to data/youtube_scraped_data.csv")
    return df

