import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from youtube_utils import fetch_youtube_videos, extract_transcripts
from summarization import summarize_transcripts
from dpr_search import setup_dpr, encode_passages, build_faiss_index, search_relevant_passages
from question_generation import generate_questions
from text_to_speech import text_to_speech_elevenlabs
import pandas as pd
from flask_session import Session
from flask import session
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
LOCAL_LLM = "llama3.2:3b-instruct-fp16"
MAX_TOKENS = 1000

# Create a Flask app instance
#app = Flask(__name__)

@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify(error=str(e)), 400

@app.errorhandler(InternalServerError)
def handle_internal_server_error(e):
    return jsonify(error="An unexpected error occurred. Please try again later."), 500

# Helper functions
def format_timestamp(seconds):
    """Convert seconds to a formatted string (MM:SS)"""
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"

def find_laughter_moments(transcript_segments, laughter_timestamps, window_size=30):
    """Find moments with high concentration of laughter"""
    laughter_moments = []
    for timestamp in laughter_timestamps:
        start = max(0, timestamp - window_size/2)
        end = timestamp + window_size/2
        relevant_segments = [seg for seg in transcript_segments if start <= seg['start'] < end]
        if relevant_segments:
            laughter_moments.append({
                'timestamp': timestamp,
                'text': ' '.join(seg['text'] for seg in relevant_segments)
            })
    return laughter_moments

# Refactored functions from main()
def fetch_and_process_videos(topic, preferences, min_views, max_results):
    videos = fetch_youtube_videos(topic, preferences, min_views, max_results)
    videos_df = pd.DataFrame(videos)
    videos_df = extract_transcripts(videos_df)
    return videos_df

def summarize_videos(videos_df):
    return summarize_transcripts(videos_df, LLM, MAX_TOKENS)

def setup_search(videos_df):
    query_encoder, query_tokenizer, passage_encoder, passage_tokenizer = setup_dpr()
    passage_embeddings = encode_passages(videos_df, passage_encoder, passage_tokenizer)
    faiss_index = build_faiss_index(passage_embeddings)
    return query_encoder, query_tokenizer, faiss_index

def perform_search(videos_df, faiss_index, query_encoder, query_tokenizer, query):
    return search_relevant_passages(
        videos_df, 
        faiss_index, 
        query_encoder, 
        query_tokenizer, 
        query, 
        top_k=5, 
        content_weight=0.7,
        laughter_weight=0.2, 
        applause_weight=0.1
    )

def prepare_results(top_k_videos):
    results = []
    for index, row in top_k_videos.iterrows():
        laughter_moments = find_laughter_moments(row['TranscriptSegments'], row['LaughterTimestamps'])
        results.append({
            'title': row['Title'],
            'link': row['Link'],
            'funny_moments': [{'timestamp': format_timestamp(moment['timestamp']), 'text': moment['text']} for moment in laughter_moments[:3]],
            'laughter_count': row['laughter_count'],
            'laughter_intensity': round(row['laughter_intensity'], 2),
            'applause_count': row['applause_count'],
            'similarity_score': round(row['Similarity Score'], 4)
        })
    return results

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
        
        # Print available columns
        app.logger.info(f"Available columns: {videos_df.columns.tolist()}")
        
        # Rename 'Views' to 'ViewCount' if it exists
        if 'Views' in videos_df.columns:
            videos_df = videos_df.rename(columns={'Views': 'ViewCount'})
        elif 'ViewCount' not in videos_df.columns:
            # If neither 'Views' nor 'ViewCount' is present, add a placeholder
            videos_df['ViewCount'] = 'N/A'
        
        app.logger.info("Setting up search")
        query_encoder, query_tokenizer, faiss_index = setup_search(videos_df)
        app.logger.info("Search setup complete")
        
        app.logger.info("Storing data in session")
        session['videos_df'] = videos_df.to_json()
        
        # Serialize FAISS index to bytes
        index_bytes = faiss.serialize_index(faiss_index)
        session['faiss_index'] = index_bytes
        
        app.logger.info("Data stored in session")
        
        videos = videos_df[['Title', 'Link', 'ViewCount']].to_dict('records')
        app.logger.info("Returning video data")
        return jsonify({'videos': videos})
    except Exception as e:
        app.logger.error(f"Error in fetch_videos: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An error occurred while fetching videos. Please try again.'}), 500


@app.route('/summarize_videos', methods=['GET'])
def summarize_videos_route():
    try:
        if 'videos_df' in session:
            videos_df = pd.read_json(session['videos_df'])
            summarized_df = summarize_videos(videos_df)
            summaries = summarized_df['Summary'].tolist()
            return jsonify({'summaries': summaries})
        else:
            return jsonify({'error': 'No videos fetched yet. Please fetch videos first.'}), 400
    except Exception as e:
        app.logger.error(f"Error in summarize_videos: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An error occurred while summarizing videos. Please try again.'}), 500


@app.route('/search_videos', methods=['POST'])
def search_videos():
    try:
        query = request.json['query']
        
        if 'videos_df' not in session or 'faiss_index' not in session:
            return jsonify({'error': 'No videos fetched or search index not set up. Please fetch videos first.'}), 400
        
        videos_df = pd.read_json(session['videos_df'])
        faiss_index = faiss.deserialize_index(session['faiss_index'])
        query_encoder, query_tokenizer, _ = setup_search(videos_df)  # We don't need to recreate the faiss_index here
        
        top_k_videos = perform_search(videos_df, faiss_index, query_encoder, query_tokenizer, query)
        results = prepare_results(top_k_videos)
        return jsonify({'results': results})
    except Exception as e:
        app.logger.error(f"Error in search_videos: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An error occurred while searching videos. Please try again.'}), 500


@app.route('/generate_questions', methods=['GET'])
def generate_questions_route():
    try:
        if 'videos_df' in session:
            videos_df = pd.read_json(session['videos_df'])
            topic = videos_df['Title'].iloc[0]  # Use the first video's title as the topic
            questions = generate_questions(topic)
            return jsonify({'questions': questions})
        else:
            return jsonify({'error': 'No videos fetched yet. Please fetch videos first.'}), 400
    except Exception as e:
        app.logger.error(f"Error in generate_questions: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An error occurred while generating questions. Please try again.'}), 500

# Run the app if this script is executed
if __name__ == '__main__':
    app.run(debug=True)