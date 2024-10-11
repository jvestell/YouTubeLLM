import os
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from youtube_utils import fetch_youtube_videos, extract_transcripts, extract_video_id
from summarization import summarize_transcripts
from similarity_search import find_preferred_moments
import pandas as pd
from flask_session import Session
import traceback
from werkzeug.exceptions import BadRequest, InternalServerError
import faiss
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'your-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Constants (you may want to move these to a config file later)
TOPIC = "Kill Tony comedy"
PREFERENCES = ["Laughter"]
MAX_RESULTS = 20
MIN_VIEWS = 100000
LLM = "facebook/bart-large-cnn"
MAX_TOKENS = 1000

@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify(error=str(e)), 400

@app.errorhandler(InternalServerError)
def handle_internal_server_error(e):
    return jsonify(error="An unexpected error occurred. Please try again later."), 500

# Refactored functions from main()
def fetch_and_process_videos(topic, preferences, min_views, max_results):
    videos = fetch_youtube_videos(topic, preferences, min_views, max_results)
    videos_df = pd.DataFrame(videos)
    videos_df = extract_transcripts(videos_df)
    return videos_df

def summarize_videos(videos_df):
    return summarize_transcripts(videos_df, LLM, MAX_TOKENS)

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/fetch_videos', methods=['POST'])
def fetch_videos():
    try:
        app.logger.info("Fetching videos started")
        topic = request.form.get('topic', TOPIC)
        preferences = request.form.getlist('preferences') or PREFERENCES
        max_results = int(request.form.get('max_results', MAX_RESULTS))
        min_views = int(request.form.get('min_views', MIN_VIEWS))

        app.logger.info(f"Fetching videos with params: topic={topic}, preferences={preferences}, max_results={max_results}, min_views={min_views}")
        videos_df = fetch_and_process_videos(topic, preferences, min_views, max_results)
        app.logger.info("Videos fetched and processed")
        
        # Rename 'Views' to 'ViewCount' if it exists
        if 'Views' in videos_df.columns:
            videos_df = videos_df.rename(columns={'Views': 'ViewCount'})
        elif 'ViewCount' not in videos_df.columns:
            # If neither 'Views' nor 'ViewCount' is present, add a placeholder
            videos_df['ViewCount'] = 'N/A'
        
        app.logger.info("Storing data in session")
        session['videos_df'] = videos_df.to_json()
        session['preferences'] = preferences  # Store preferences in session
        
        app.logger.info("Data stored in session")
        
        videos = videos_df[['Title', 'Link', 'ViewCount']].to_dict('records')
        app.logger.info("Returning video data")
        return jsonify({'videos': videos})
    except Exception as e:
        app.logger.error(f"Error in fetch_videos: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An error occurred while fetching videos. Please try again.'}), 500

@app.route('/show_preferred_moments', methods=['GET'])
def show_preferred_moments():
    try:
        if 'videos_df' in session:
            videos_df = pd.read_json(session['videos_df'])
            preferences = session.get('preferences', PREFERENCES)
            preferred_moments = find_preferred_moments(videos_df, preferences)
            if not preferred_moments:
                return jsonify({'error': 'No preferred moments found. Try adjusting your preferences or fetching more videos.'}), 404
            return jsonify({'preferred_moments': preferred_moments})
        else:
            return jsonify({'error': 'No videos fetched yet. Please fetch videos first.'}), 400
    except Exception as e:
        app.logger.error(f"Error in show_preferred_moments: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An error occurred while showing preferred moments. Please try again.'}), 500

# Run the app if this script is executed
if __name__ == '__main__':
    app.run(debug=True)