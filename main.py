import os
from dotenv import load_dotenv
from youtube_utils import fetch_youtube_videos, display_videos, extract_transcripts
from summarization import summarize_transcripts
from dpr_search import setup_dpr, encode_passages, build_faiss_index, search_relevant_passages
from question_generation import generate_questions, display_questions_with_markdown
from text_to_speech import text_to_speech_elevenlabs

# Load environment variables
load_dotenv()

# Constants
TOPIC = "Kill Tony comedy"
PREFERENCES = ["Laughter"]
MAX_RESULTS = 20
MIN_VIEWS = 100000
LLM = "facebook/bart-large-cnn"
LOCAL_LLM = "llama3.2:3b-instruct-fp16"
MAX_TOKENS = 1000

def main():
    # 1. Fetch YouTube videos
    print("Fetching relevant YouTube videos about comedy...\n")
    videos = fetch_youtube_videos(TOPIC, PREFERENCES, MIN_VIEWS, MAX_RESULTS)
    videos_df = display_videos(videos)

    # 2. Extract transcripts
    videos_df = extract_transcripts(videos_df)

    # 3. Summarize transcripts
    videos_df = summarize_transcripts(videos_df, LLM, MAX_TOKENS)

    # 4. Set up DPR and search
    query_encoder, query_tokenizer, passage_encoder, passage_tokenizer = setup_dpr()
    passage_embeddings = encode_passages(videos_df, passage_encoder, passage_tokenizer)
    faiss_index = build_faiss_index(passage_embeddings)

    # Example query
    query = "What are the best jokes that create laughter?"
    top_k_videos = search_relevant_passages(videos_df, faiss_index, query_encoder, query_tokenizer, query, top_k=5)
    print("\nTop 5 Jokes Found:")
    for index, row in top_k_videos.iterrows():
        print(f"\nJoke from video: '{row['Title']}'")
        print(f"Transcript excerpt: {row['Transcript'][:300]}...")  # Print first 200 characters of the transcript
        print(f"Similarity Score: {row['Similarity Score']:.2f}")
        print("-" * 50)
    #print("Filtered Top-k Videos with Similarity Score and Query:")
    #print(top_k_videos[['Title', 'Transcript', 'Similarity Score', 'Query']])

    # 5. Generate questions
    display_questions_with_markdown(TOPIC)

    # 6. Text to speech
    text_to_speak = '''Top 3 Questions for First-Time Travelers to Amsterdam:
    Here are the top 3 questions that a first-time traveler might ask about visiting Amsterdam:

    What is the best way to get around Amsterdam?
    Is Amsterdam safe for tourists?
    What are some must-see attractions in Amsterdam?
    '''
    #result = text_to_speech_elevenlabs(text_to_speak)
    #print(result)

if __name__ == "__main__":
    main()