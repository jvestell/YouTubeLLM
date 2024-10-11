from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

def find_preferred_moments(videos_df, preferences, top_k=5):
    preferred_moments = []
    
    for _, row in videos_df.iterrows():
        transcript = row['Transcript']
        video_id = extract_video_id(row['Link'])
        
        if not isinstance(transcript, str) or transcript.startswith("Error") or transcript.startswith("No transcript"):
            logging.warning(f"Skipping video {video_id} due to missing or invalid transcript")
            continue

        # Split transcript into sentences
        sentences = transcript.split('.')
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        sentence_vectors = vectorizer.fit_transform(sentences)
        
        # Create preference vector
        preference_vector = vectorizer.transform([' '.join(preferences)])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(preference_vector, sentence_vectors)
        
        # Get top k similar sentences
        top_indices = similarities.argsort()[0][-top_k:][::-1]
        
        for idx in top_indices:
            timestamp = get_timestamp(row['TranscriptSegments'], sentences[idx])
            preferred_moments.append({
                'video_id': video_id,
                'timestamp': timestamp,
                'text': sentences[idx].strip(),
                'similarity_score': float(similarities[0][idx])
            })
            logging.info(f"Found preferred moment: video_id={video_id}, timestamp={timestamp}, text={sentences[idx].strip()}")
    
    # Sort all moments by similarity score
    preferred_moments.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return preferred_moments[:top_k]

def get_timestamp(transcript_segments, sentence):
    sentence = sentence.strip().lower()
    if not sentence or not transcript_segments:
        return 0  # Return start of video if sentence is empty or no transcript segments

    best_match = None
    best_match_ratio = 0

    sentence_words = set(sentence.split())
    if not sentence_words:
        return 0  # Return start of video if sentence has no words

    for segment in transcript_segments:
        segment_text = segment['text'].lower()
        segment_words = set(segment_text.split())
        
        # Calculate the ratio of words from the sentence that appear in the segment
        if sentence_words:
            match_ratio = len(sentence_words.intersection(segment_words)) / len(sentence_words)
        else:
            match_ratio = 0

        if match_ratio > best_match_ratio:
            best_match = segment
            best_match_ratio = match_ratio

        # If we find an exact match or a very close match, return immediately
        if match_ratio > 0.8:
            return segment['start']

    # If we found any match at all, return its timestamp
    if best_match:
        return best_match['start']

    # If no match found, return the start of the video
    return 0

def extract_video_id(url):
    # Extract video ID from YouTube URL
    video_id = url.split('v=')[-1]
    ampersand_pos = video_id.find('&')
    if ampersand_pos != -1:
        video_id = video_id[:ampersand_pos]
    return video_id