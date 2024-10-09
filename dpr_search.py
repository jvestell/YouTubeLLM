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

def search_relevant_passages(videos_df, faiss_index, query_encoder, query_tokenizer, query, top_k=5, laughter_weight=0.1, applause_weight=0.05):
    """
    Encode the query and search for the most relevant passages, incorporating laughter and applause counts
    """
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
    base_similarity_scores = 1 / (1 + np.exp(distances[0]))  # Convert distances to similarity scores
    
    # Normalize laughter and applause counts
    max_laughter = candidate_videos['laughter_count'].max()
    max_applause = candidate_videos['applause_count'].max()
    normalized_laughter = candidate_videos['laughter_count'] / max_laughter if max_laughter > 0 else 0
    normalized_applause = candidate_videos['applause_count'] / max_applause if max_applause > 0 else 0
    
    # Calculate the final similarity scores
    content_weight = 1 - (laughter_weight + applause_weight)
    final_similarity_scores = (
        base_similarity_scores * content_weight +
        normalized_laughter * laughter_weight +
        normalized_applause * applause_weight
    )
    
    # Sort and select top_k videos based on the final similarity scores
    candidate_videos['Similarity Score'] = final_similarity_scores
    top_k_videos = candidate_videos.nlargest(top_k, 'Similarity Score')
    
    # Add query information
    top_k_videos['Query'] = query

    return top_k_videos
