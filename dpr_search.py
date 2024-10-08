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

def search_relevant_passages(videos_df, faiss_index, query_encoder, query_tokenizer, query, top_k=3):
    """
    Encode the query and search for the most relevant passages
    """
    query_inputs = query_tokenizer(query, return_tensors='pt', max_length=128, truncation=True, padding=True)

    with torch.no_grad():
        query_embedding = query_encoder(**query_inputs).pooler_output.numpy()

    query_embedding = query_embedding.reshape(1, -1)

    distances, indices = faiss_index.search(query_embedding, top_k)

    top_k_indices = indices[0][:top_k]
    top_k_videos = videos_df.iloc[top_k_indices].copy()
    top_k_videos['Similarity Score'] = distances[0][:top_k]
    top_k_videos['Query'] = query

    return top_k_videos
