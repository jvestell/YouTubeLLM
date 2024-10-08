import re
from transformers import AutoTokenizer, pipeline

def clean_text(text):
    """
    Cleans the input text by removing non-ASCII characters and extra whitespace.
    """
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_text_into_chunks(text, tokenizer, max_tokens):
    """
    Splits the input text into chunks based on the maximum token limit.
    """
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def summarize_text(transcript, summarizer_pipeline):
    """
    Summarizes the provided transcript using the BART model.
    """
    try:
        t = clean_text(transcript)
        chunks = split_text_into_chunks(t, summarizer_pipeline.tokenizer, summarizer_pipeline.model.config.max_position_embeddings)
        print(f"Nr of chunks: {len(chunks)}")

        summaries = [summarizer_pipeline(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
        result = ' '.join(summaries)

        return result if len(result) > 0 else "No summary generated."
    except Exception as e:
        return f"Error summarizing transcript: {e}"

def summarize_transcripts(videos_df, llm_model, max_tokens):
    """
    Summarizes transcripts for all videos in the DataFrame.
    """
    summarizer_pipeline = pipeline("summarization", model=llm_model)
    videos_df['Summary'] = None

    for index, row in videos_df.iterrows():
        transcript = row['Transcript']
        video_title = row['Title']

        if transcript and "transcripts are disabled" not in transcript.lower() and "no transcripts found" not in transcript.lower():
            print(f"Summarizing transcript for video: '{video_title}'")
            try:
                summary = summarize_text(transcript, summarizer_pipeline)
                videos_df.at[index, 'Summary'] = summary
                print(f"Summary generated for video: '{video_title}'\n")
            except Exception as e:
                print(f"An error occurred while summarizing video: '{video_title}'. Error: {e}\n")
                videos_df.at[index, 'Summary'] = f"Error summarizing transcript: {e}"
        else:
            print(f"No valid transcript found for video: '{video_title}'. Skipping summarization.\n")
            videos_df.at[index, 'Summary'] = "No transcript available."

    return videos_df