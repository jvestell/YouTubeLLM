from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_preferred_moments(videos_df, preferences, top_k=5):
    preferred_moments = []
    
    for _, row in videos_df.iterrows():
        transcript = row['Transcript']
        video_id = extract_video_id(row['Link'])
        
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
    
    # Sort all moments by similarity score
    preferred_moments.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return preferred_moments[:top_k]

def get_timestamp(transcript_segments, sentence):
    for segment in transcript_segments:
        if sentence.strip() in segment['text']:
            return segment['start']
    return 0  # Default to start of video if no match found

def extract_video_id(url):
    # Extract video ID from YouTube URL
    video_id = url.split('v=')[-1]
    ampersand_pos = video_id.find('&')
    if ampersand_pos != -1:
        video_id = video_id[:ampersand_pos]
    return video_id