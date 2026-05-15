# Date Created: 2025-09-20 17:23:40+03:00
# Date Last Modified: 2026-05-15

""" 02) perfume_emotion_scraper.py
Purpose: Scrapes perfume review data from YouTube or another source.
Input: None or external queries/URLs
Output: YouTube_Perfume_Reviews_Partial.csv or similar
Execution Order: Manual first step """

import os
import requests
import pandas as pd
import time
import re
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from src.utils.paths import data_path

# ------------------------------ CONFIG ------------------------------
BATCH_SIZE = 95
POPULAR_BRANDS = [
    "chanel", "dior", "versace", "armani", "ysl", "tom ford", "gucci",
    "lancome", "burberry", "jo malone", "byredo", "creed", "le labo", "narciso", 
    "zara", "marc jacobs", "maison francis kurkdjian", "mugler", "valentino"
]
EMOTION_KEYWORDS = set([
    "love", "hate", "beautiful", "fresh", "intoxicating", "romantic", "cozy", "sensual", 
    "clean", "dark", "bright", "sexy", "calm", "uplifting", "warm", "inviting",
    "alluring", "boring", "powerful", "peaceful", "happy", "sad", "serene", "energetic"
])
# -------------------------------------------------------------------

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY") or "YOUR_API_KEY_HERE"

def score_perfume(name):
    return any(brand in name.lower() for brand in POPULAR_BRANDS)

# Load perfume list
df_perfumes = pd.read_csv(data_path("perfume_level", "final_perfume_data.csv"), encoding="ISO-8859-1")
df_perfumes["Score"] = df_perfumes.iloc[:, 0].apply(score_perfume)
df_perfumes = df_perfumes.sort_values(by="Score", ascending=False)
all_perfumes = df_perfumes.iloc[:, 0].dropna().astype(str).str.lower().tolist()

# Load progress logs
done_log_file = data_path("intermediate", "processed_perfumes.csv")
# perfume_emotion_scraper.py
output_file = data_path("raw", "YouTube_Perfume_Reviews_Partial.csv")

if done_log_file.exists():
    processed = set(pd.read_csv(done_log_file)["Perfume"].str.lower())
else:
    processed = set()

# Filter perfumes
to_process = [p for p in all_perfumes if p not in processed][:BATCH_SIZE]
print(f"📦 Processing {len(to_process)} perfumes today...")

def search_youtube(query):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "maxResults": 10,
        "type": "video",
        "key": API_KEY,
    }
    response = requests.get(url, params=params)
    return response.json()

def get_youtube_transcript(video_id, max_retries=3):
    for _ in range(max_retries):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            return " ".join([entry["text"] for entry in transcript])
        except (TranscriptsDisabled, NoTranscriptFound):
            return ""
        except Exception:
            time.sleep(1)
    return ""

def emotion_keyword_count(text):
    return sum(1 for word in text.lower().split() if word in EMOTION_KEYWORDS)

def fallback_search(perfume):
    queries = [f"{perfume} perfume", f"{perfume} fragrance", f"{perfume} review"]
    for query in queries:
        print(f"🔍 Searching: {query}")
        data = search_youtube(query)
        results = []
        for item in data.get("items", []):
            results.append({
                "Perfume": perfume,
                "Video ID": item["id"]["videoId"],
                "Title": item["snippet"]["title"],
                "Description": item["snippet"]["description"],
                "URL": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "Channel Title": item["snippet"]["channelTitle"],
            })
        if results:
            print(f"✅ Found {len(results)} videos for: {query}")
            return results
        else:
            print(f"⚠️ No results for: {query}")
    return []

def fetch_transcript(video):
    video["Transcript"] = get_youtube_transcript(video["Video ID"])
    return video

# Process each perfume
all_videos = []
perfume_status = []
for perfume in to_process:
    print(f"🚀 Processing perfume: {perfume}")
    videos = fallback_search(perfume)
    if not videos:
        perfume_status.append({"Perfume": perfume, "Status": "No videos"})
        continue
    time.sleep(1)
    with ThreadPoolExecutor(max_workers=10) as executor:
        videos = list(executor.map(fetch_transcript, videos))

    valid = []
    for video in videos:
        txt = video.get("Transcript", "")
        word_count = len(txt.split())
        emotion_count = emotion_keyword_count(txt)
        if word_count >= 20:
            video["Transcript Length"] = word_count
            video["Emotion Keyword Count"] = emotion_count
            video["Final Transcript"] = txt
            valid.append(video)

    if valid:
        print(f"✅ {len(valid)} usable transcripts found for: {perfume}")
        all_videos.extend(valid)
        perfume_status.append({"Perfume": perfume, "Status": "Done"})
    else:
        print(f"❌ No usable transcripts for: {perfume}")
        perfume_status.append({"Perfume": perfume, "Status": "No results"})

# Save to output
if all_videos:
    df_new = pd.DataFrame(all_videos)
    if output_file.exists():
        df_old = pd.read_csv(output_file)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(output_file, index=False)
    print(f"📀 Saved {len(df_new)} new entries to {output_file}")
else:
    print("📬 No valid videos found today.")

# Update progress log
df_log = pd.DataFrame(perfume_status)
if done_log_file.exists():
    old_log = pd.read_csv(done_log_file)
    new_log = pd.concat([old_log, df_log], ignore_index=True).drop_duplicates()
else:
    new_log = df_log
new_log.to_csv(done_log_file, index=False)
print(f"📋 Updated log with {len(df_log)} perfumes.")
