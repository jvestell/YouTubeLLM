import torch
import numpy as np
import faiss
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer

def setup_dpr():
    """
    Set up DPR models (Query Encoder and Passage Encoder)
    """
    query_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    passage_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    passage_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    
    return query_encoder, query_tokenizer, passage_encoder, passage_tokenizer

def encode_passages(videos_df, passage_encoder, passage_tokenizer):
    """
    Encode passages (transcripts) using the DPR context encoder
    """
    passages = videos_df['Transcript'].tolist()
    passage_embeddings = []

    for passage in passages:
        inputs = passage_tokenizer(passage, return_tensors='pt', max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            embedding = passage_encoder(**inputs).pooler_output
        passage_embeddings.append(embedding.numpy())

    return np.vstack(passage_embeddings)

def build_faiss_index(passage_embeddings):
    """
    Build a FAISS index for fast retrieval
    """
    dimension = passage_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(passage_embeddings)
    return faiss_index

def calculate_laughter_intensity(laughter_timestamps, video_duration, window_size=30):
    """
    Calculate laughter intensity based on the frequency of laughter in a sliding window.
    
    :param laughter_timestamps: List of timestamps where laughter occurred
    :param video_duration: Total duration of the video in seconds
    :param window_size: Size of the sliding window in seconds
    :return: A laughter intensity score
    """
    if not laughter_timestamps:
        return 0
    
    # Create a histogram of laughter occurrences
    num_bins = int(video_duration / window_size) + 1
    hist, _ = np.histogram(laughter_timestamps, bins=num_bins, range=(0, video_duration))
    
    # Calculate the intensity score
    # We'll use the maximum number of laughs in any window as our intensity score
    intensity_score = np.max(hist)
    
    # Normalize the score by dividing by the window size
    normalized_intensity = intensity_score / window_size
    
    return normalized_intensity

def get_video_duration(transcript_segments):
    if transcript_segments and isinstance(transcript_segments, list) and transcript_segments[-1]:
        last_segment = transcript_segments[-1]
        return last_segment['start'] + last_segment['duration']
    return 0  # Return 0 if no valid transcript segments

def search_relevant_passages(videos_df, faiss_index, query_encoder, query_tokenizer, query, top_k=5, content_weight=0.7, laughter_weight=0.2, applause_weight=0.1):

    # Deserialize the FAISS index if it's in bytes format
    if isinstance(faiss_index, bytes):
        faiss_index = faiss.deserialize(faiss_index)

    query_inputs = query_tokenizer(query, return_tensors='pt', max_length=128, truncation=True, padding=True)

    with torch.no_grad():
        query_embedding = query_encoder(**query_inputs).pooler_output.numpy()

    query_embedding = query_embedding.reshape(1, -1)

    # Initial search using FAISS
    distances, indices = faiss_index.search(query_embedding, top_k * 2)  # Search for more candidates initially

    # Get the candidate videos
    candidate_indices = indices[0]
    candidate_videos = videos_df.iloc[candidate_indices].copy()
    
    # Calculate the base similarity scores
    base_similarity_scores = 1 / (1 + np.exp(-distances[0]))  # Convert distances to similarity scores
    
    # Calculate laughter intensity for each video
    candidate_videos['laughter_intensity'] = candidate_videos.apply(
        lambda row: calculate_laughter_intensity(
            row['LaughterTimestamps'] if isinstance(row['LaughterTimestamps'], list) else [],
            get_video_duration(row['TranscriptSegments'])
        ),
        axis=1
    )
    
    # Normalize laughter intensity and applause counts
    max_laughter_intensity = candidate_videos['laughter_intensity'].max()
    max_applause = candidate_videos['applause_count'].max()
    normalized_laughter_intensity = candidate_videos['laughter_intensity'] / max_laughter_intensity if max_laughter_intensity > 0 else 0
    normalized_applause = candidate_videos['applause_count'] / max_applause if max_applause > 0 else 0
    
    # Calculate the final similarity scores
    final_similarity_scores = (
        base_similarity_scores * content_weight +
        normalized_laughter_intensity * laughter_weight +
        normalized_applause * applause_weight
    )
    
    # Sort and select top_k videos based on the final similarity scores
    candidate_videos['Similarity Score'] = final_similarity_scores
    top_k_videos = candidate_videos.nlargest(top_k, 'Similarity Score')
    
    # Add query information
    top_k_videos['Query'] = query

    return top_k_videos
