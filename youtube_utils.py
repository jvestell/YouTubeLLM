import re
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import pandas as pd

def parse_views(views_str):
    """
    Parses the views string from YouTube and converts it to an integer.
    """
    views_str = views_str.lower().replace('views', '').replace(',', '').strip()
    return int(views_str)

def fetch_youtube_videos(destination, preferences, min_views, max_results=10):
    """
    Fetches relevant YouTube videos based on the destination and user preferences.
    """
    preferences_query = ' '.join(preferences)
    search_query = f"{destination} travel guide {preferences_query} Netherlands"
    videos_search = VideosSearch(search_query, limit=max_results)
    search_results = videos_search.result()

    videos = []
    for video in search_results['result']:
        views = parse_views(video['viewCount']['text'])
        if views < min_views:
            continue
        video_data = {
            'Title': video['title'],
            'Duration': video['duration'],
            'Views': views,
            'Channel': video['channel']['name'],
            'Link': video['link']
        }
        videos.append(video_data)
    return videos

def display_videos(videos):
    """
    Displays the list of videos in a pandas DataFrame.
    """
    if not videos:
        print("No videos found with more than 10,000 views.")
        return pd.DataFrame()

    videos_df = pd.DataFrame(videos)
    print("\nFetched YouTube Videos (Filtered by >10,000 views):")
    print(videos_df[['Title', 'Duration', 'Views', 'Channel']].to_string(index=False))
    return videos_df

def extract_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    """
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
    return video_id_match.group(1) if video_id_match else None

def extract_transcripts(videos_df):
    """
    Extracts transcripts for each video in the DataFrame.
    """
    for index, row in videos_df.iterrows():
        youtube_url = row['Link']
        video_title = row['Title']
        video_id = extract_video_id(youtube_url)

        if video_id:
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript = ' '.join([segment['text'] for segment in transcript_list])
                videos_df.at[index, 'Transcript'] = transcript
                print(f"Transcript fetched for video: '{video_title}'")
            except TranscriptsDisabled:
                videos_df.at[index, 'Transcript'] = "Transcripts are disabled for this video."
            except NoTranscriptFound:
                videos_df.at[index, 'Transcript'] = "No transcripts found for this video."
            except Exception as e:
                videos_df.at[index, 'Transcript'] = f"Error fetching transcript: {e}"
        else:
            videos_df.at[index, 'Transcript'] = "Invalid YouTube URL."
    
    return videos_df