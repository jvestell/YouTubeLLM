import re
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import pandas as pd
import webbrowser

def parse_views(views_str):
    """
    Parses the views string from YouTube and converts it to an integer.
    """
    if len(views_str) > 1:
        views_str = views_str.lower().replace('views', '').replace(',', '').strip()
    return int(views_str)

def fetch_youtube_videos(topic, preferences, min_views, max_results=10):
    """
    Fetches relevant YouTube videos based on the destination and user preferences.
    """
    preferences_query = ' '.join(preferences)
    search_query = f"{topic} episodes that mention {preferences_query}"
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
    Displays the list of videos in a pandas DataFrame and provides an option to open videos in a web browser.
    """
    if not videos:
        print("No videos found with more than 1,000,000 views.")
        return pd.DataFrame()

    videos_df = pd.DataFrame(videos)
    print("\nFetched YouTube Videos (Filtered by >1,000,000 views):")
    print(videos_df[['Title', 'Duration', 'Views', 'Channel']].to_string(index=False))
    
    while True:
        choice = input("\nEnter the number of the video you want to open (1-{}) or 'q' to quit: ".format(len(videos)))
        if choice.lower() == 'q':
            break
        try:
            index = int(choice) - 1
            if 0 <= index < len(videos):
                url = videos_df.iloc[index]['Link']
                print(f"Opening: {videos_df.iloc[index]['Title']}")
                webbrowser.open(url)
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
    
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
    
    videos_df['laughter_count'] = videos_df['Transcript'].apply(lambda x: x.count('[Laughter]'))
    videos_df['applause_count'] = videos_df['Transcript'].apply(lambda x: x.count('[Applause]'))
    return videos_df