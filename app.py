
import re
import os
import webbrowser
import numpy as np
import pandas as pd
from IPython.display import display, HTML

# DL
import torch

# Youtube
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Huggingface
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,  AutoModelForSeq2SeqLM
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer

# Vector Databases
import faiss


# Langchin | Elevenlabs | Langchain Agents
from langchain.chains import LLMChain
#from langchain import langchain.chains.LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.eleven_labs.text2speech import ElevenLabsText2SpeechTool
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import markdown
import requests
import json
load_dotenv()
"""## Load Constants"""

# Constants (Predefined Inputs)
DESTINATION = "Amsterdam"
PREFERENCES = ["Museums", "Outdoor Activities"]
MAX_RESULTS = 20  # Number of videos to fetch
MIN_VIEWS = 10000  # 10,000 views # Threshold for minimum views
LLM = "facebook/bart-large-cnn"
LOCAL_LLM = "llama3.2:3b-instruct-fp16"
MAX_TOKENS = 1000

"""# Build PipeLine Step by Step

## **1.Search YouTube Videos**
"""

def parse_views(views_str):
    """
    Parses the views string from YouTube and converts it to an integer.

    5,909 views -> 5909

    """
    # Remove the "views" part and any commas
    views_str = views_str.lower().replace('views', '').replace(',', '').strip()


    return int(views_str)  # In case of any parsing error

def fetch_youtube_videos(destination, preferences, MIN_VIEWS, max_results=10):
    """
    Fetches relevant YouTube videos based on the destination and user preferences.

    Parameters:
    - destination (str): The travel destination (e.g., 'Amsterdam').
    - preferences (list): List of user-selected preferences (e.g., ['Museums', 'Outdoor Activities']).
    - max_results (int): Maximum number of videos to fetch.

    Returns:
    - videos (list): List of dictionaries containing video details with views parsed as integers.
    """
    # Combine preferences into a search query
    preferences_query = ' '.join(preferences)
    search_query = f"{destination} travel guide {preferences_query} Netherlands"

    # Initialize VideosSearch
    videos_search = VideosSearch(search_query, limit=max_results)

    # Execute search
    search_results = videos_search.result()

    videos = []
    for video in search_results['result']:
        views_str = video['viewCount']['text']
        views = parse_views(views_str)
        print(f"parseview {views}")

        # Filter out videos with fewer than MIN_VIEWS
        if views < MIN_VIEWS:
            continue  # Skip this video

        video_data = {
            'Title': video['title'],
            'Duration': video['duration'],
            'Views': views,  # Store as integer
            'Channel': video['channel']['name'],
            'Link': video['link']
        }
        videos.append(video_data)

    return videos

print("Fetching relevant YouTube videos about traveling in Amsterdam...\n")
videos = fetch_youtube_videos(DESTINATION, PREFERENCES, MIN_VIEWS, MAX_RESULTS)

len(videos)

videos

def display_videos(videos):
    """
    Displays the list of videos in a pandas DataFrame and optionally opens them in the browser.

    Parameters:
    - videos (list): List of dictionaries containing video details.
    """
    if not videos:
        print("No videos found with more than 10,000 views.")
        return

    # Create a DataFrame for better display
    videos_df = pd.DataFrame(videos)
    print("\nFetched YouTube Videos (Filtered by >10,000 views):")
    print(videos_df[['Title', 'Duration', 'Views', 'Channel']].to_string(index=False))

    # Optionally, ask the user if they want to open the videos in the browser
    open_browser = input("\nDo you want to open these videos in your browser? (y/n): ").strip().lower()
    if open_browser == 'y':
        for video in videos:
            webbrowser.open(video['Link'])

display_videos(videos)

# We can not open Youtube Videos in Webbrowser via Google Colab
webbrowser.open("https://www.youtube.com/watch?v=zFABm07RtXk")

# Display clickable links
print("\n### Watch These Videos:")
for idx, video in enumerate(videos, 1):
        # Display as markdown link
        display(HTML(f"{idx}. <a href='{video['Link']}' target='_blank'>{video['Title']}</a>"))

def display_videos(videos):
    """
    Displays the list of videos in a pandas DataFrame and embeds them in the notebook.

    Parameters:
    - videos (list): List of dictionaries containing video details.
    """
    if not videos:
        print("No videos found with more than 10,000 views.")
        return

    # Create a DataFrame for better display
    videos_df = pd.DataFrame(videos)
    print("\nFetched YouTube Videos (Filtered by >10,000 views):")
    print(videos_df[['Title', 'Duration', 'Views', 'Channel']].to_string(index=False))

    # Display clickable links
    print("\n### Watch These Videos:")
    for idx, video in enumerate(videos, 1):
        # Display as markdown link
        display(HTML(f"{idx}. <a href='{video['Link']}' target='_blank'>{video['Title']}</a>"))
    return videos_df

videos_df = display_videos(videos)

videos_df

"""## **2.Transcript YouTube Videos**"""

def extract_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.

    Parameters:
    - youtube_url (str): The full YouTube video URL.

    Returns:
    - video_id (str): The extracted video ID.
    """
    # Regular expression to extract video ID
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        return None

# Example
extract_video_id("https://www.youtube.com/watch?v=abcd1234EFG")
# Output: 'abcd1234EFG'

def extract_transcripts(videos_df):
  # Initialize a new column for transcripts
  videos_df['Transcript'] = None

  # Iterate over each video and fetch the transcript
  for index, row in videos_df.iterrows():
      youtube_url = row['Link']
      video_title = row['Title']
      video_id = extract_video_id(youtube_url)

      if video_id:
          try:
              # Fetch the transcript using the video ID
              transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

              # Combine the transcript segments into a single string
              transcript = ' '.join([segment['text'] for segment in transcript_list])

              # Assign the transcript to the DataFrame
              videos_df.at[index, 'Transcript'] = transcript
              print(f"Transcript fetched for video: '{video_title}'")

          except TranscriptsDisabled:
              print(f"Transcripts are disabled for video: '{video_title}'.")
              videos_df.at[index, 'Transcript'] = "Transcripts are disabled for this video."

          except NoTranscriptFound:
              print(f"No transcripts found for video: '{video_title}'.")
              videos_df.at[index, 'Transcript'] = "No transcripts found for this video."

          except Exception as e:
              print(f"An error occurred while fetching transcript for video: '{video_title}'. Error: {e}")
              videos_df.at[index, 'Transcript'] = f"Error fetching transcript: {e}"
      else:
          print(f"Could not extract video ID from URL: {youtube_url}")
          videos_df.at[index, 'Transcript'] = "Invalid YouTube URL."
  return videos_df


videos_df = extract_transcripts(videos_df)

videos_df

"""## **3.Summerization Using Facebook LLM via Hugging Face**"""

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM)
summarizer = pipeline("summarization", model=LLM)

# Clean text function
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_text_into_chunks(text, max_tokens=MAX_TOKENS):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# Split the text into chunks
t = clean_text(videos_df['Transcript'][0])
chunks = split_text_into_chunks(t)

# Summarize each chunk
summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]

# Combine the summaries if needed
final_summary = ' '.join(summaries)
final_summary

len(t)

tokens = tokenizer.encode(t)
print(f"Number of tokens: {len(tokens)}")

t

videos_df.iloc[0:1]

def split_text_into_chunks(text, max_tokens=MAX_TOKENS):
    tokenizer = AutoTokenizer.from_pretrained(LLM)
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def summarize_text(transcript, summarizer_pipeline):
    """
    Summarizes the provided transcript using the BART model.

    Parameters:
    - transcript (str): The video transcript to summarize.

    Returns:
    - summary (str): The summarized text or an error message.
    """
    try:

        # Split the text into chunks
        t = clean_text(transcript)
        chunks = split_text_into_chunks(t)
        print(f"Nr of chuncks: {len(chunks)}")

        # Summarize each chunk
        summaries = [summarizer_pipeline(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]

        # Combine the summaries if needed
        result = ' '.join(summaries)

        if len(result) > 0:
            return result
        else:
            return "No summary generated."
    except Exception as e:
        # Return an error message in case of failure
        return f"Error summarizing transcript: {e}"


# Initialize a new column for summaries
videos_df['Summary'] = None
summarizer_pipeline = pipeline("summarization", model=LLM)

    # Iterate over each transcript and generate summaries
for index, row in videos_df.iloc[0:1].iterrows():
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

# Display the updated DataFrame with summaries
videos_df

summary

"""## 4.Building Dense Passage Retrieval (DPR)
where we can efficiently retrieve the most relevant parts of the transcripts related to travel in Amsterdam. After that, we can process or summarize the retrieved passages.

### 4.1 Set Up DPR Models (Query Encoder and Passage Encoder)

Hugging Face provides DPR models with two main components:
Query Encoder: Encodes the input query.
Passage Encoder: Encodes the passages to be searched.
"""

# Load DPR question encoder and tokenizer for the query
query_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# Load DPR context encoder and tokenizer for the passages (i.e., transcript sections)
passage_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
passage_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

"""### 4.2 Encode Passages (Transcripts)
We will encode the passages (i.e., parts of the transcript) and store their embeddings for efficient retrieval.
"""

# Encode passages (transcripts) using the DPR context encoder
def encode_passages(videos_df):
    passages = videos_df['Transcript'].tolist()
    passage_embeddings = []

    for passage in passages:
        inputs = passage_tokenizer(passage, return_tensors='pt', max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            embedding = passage_encoder(**inputs).pooler_output
        passage_embeddings.append(embedding.numpy())

    # Convert to a numpy array
    passage_embeddings = np.vstack(passage_embeddings)

    return passage_embeddings

# Encode all the transcripts in the DataFrame
passage_embeddings = encode_passages(videos_df)

passage_embeddings

"""### 4.3 Build a FAISS Index for Fast Retrieval

We'll use FAISS to index the encoded passages, making it faster to retrieve relevant passages based on the query.
"""

# Initialize FAISS index for similarity search
dimension = passage_embeddings.shape[1]  # DPR embedding size is 768
faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (dot product) for similarity

# Add the passage embeddings to the index
faiss_index.add(passage_embeddings)

faiss_index

"""### 4.4 Encode the Query and Retrieve Relevant Passages
Next, we will encode the user's query and use it to retrieve the most relevant passages from the indexed transcripts.
"""

# Function to encode the query and search for the most relevant passages
def search_relevant_passages(query, faiss_index, top_k=3):
    # Encode the query using the DPR question encoder
    query_inputs = query_tokenizer(query, return_tensors='pt', max_length=128, truncation=True, padding=True)

    with torch.no_grad():
        query_embedding = query_encoder(**query_inputs).pooler_output.numpy()

    # Ensure query_embedding is 2D (i.e., shape (1, embedding_dimension))
    query_embedding = query_embedding.reshape(1, -1)  # Reshaping to (1, embedding_dimension)

    # Search for the top-k most similar passages
    distances, indices = faiss_index.search(query_embedding, top_k)

    print(f"distances: {distances}")
    print(f"indices: {indices}")

    # Retrieve the corresponding transcripts and titles
    retrieved_passages = []
    for i in range(top_k):  # Loop through top_k results
        idx = indices[0][i]  # Get the index of the passage
        retrieved_passages.append({
            'Title': videos_df.iloc[idx]['Title'],  # Get the title using the index
            'Transcript': videos_df.iloc[idx]['Transcript'],  # Get the transcript using the index
            'Similarity Score': distances[0][i]  # Get the corresponding similarity score using i, not idx
        })

    return retrieved_passages

# Example query
query = "What are the best museums in Amsterdam?"

# Search for the most relevant passages
retrieved_passages = search_relevant_passages(query, faiss_index, top_k=2)

# Display retrieved results
for passage in retrieved_passages:
    print(f"Title: {passage['Title']}")
    print(f"Transcript: {passage['Transcript']}")
    print(f"Similarity Score: {passage['Similarity Score']}\n")

# Function to encode the query and search for the most relevant passages
def search_relevant_passages(videos_df, faiss_index, query, top_k=3):
    # Encode the query using the DPR question encoder
    query_inputs = query_tokenizer(query, return_tensors='pt', max_length=128, truncation=True, padding=True)

    with torch.no_grad():
        query_embedding = query_encoder(**query_inputs).pooler_output.numpy()

    # Ensure query_embedding is 2D (i.e., shape (1, embedding_dimension))
    query_embedding = query_embedding.reshape(1, -1)  # Reshaping to (1, embedding_dimension)

    # Search for the top-k most similar passages
    distances, indices = faiss_index.search(query_embedding, top_k)

    # Filter the DataFrame based on the retrieved indices
    top_k_indices = indices[0][:top_k]  # Get the top_k indices from the FAISS search result
    top_k_videos = videos_df.iloc[top_k_indices].copy()  # Use iloc to filter the DataFrame based on these indices and create a copy

    # Add the similarity score as a new column in the DataFrame
    top_k_videos['Similarity Score'] = distances[0][:top_k]

    # Add the query as a new column in the DataFrame
    top_k_videos['Query'] = query

    return top_k_videos

# Example query
query = "What are the best museums in Amsterdam?"

# Search for the most relevant passages and filter videos from videos_df
top_k_videos = search_relevant_passages(videos_df, faiss_index, query, top_k=3)

# Display the top_k filtered videos along with the similarity scores and query
print("Filtered Top-k Videos with Similarity Score and Query:")
top_k_videos[['Title', 'Transcript', 'Similarity Score', 'Query']]

"""## 5. Building Travel Agent to Generate Top 10 Queries"""


# Function to generate questions using ChatLLaMA from Ollama
def generate_questions(city):
    prompt = f"Act as an Travel Agent and Expert in {city} tour Guide. Generate a list of the top 10 questions that a first-time traveler might ask about visiting {city}."

    response = requests.post('http://localhost:11434/api/chat', 
                             json={
                                 'model': 'llama3.2:3b-instruct-fp16',
                                 'messages': [{"role": "user", "content": prompt}]
                             },
                             stream=True)  # Enable streaming

    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_response = json.loads(line)
                if 'response' in json_response:
                    full_response += json_response['response']
                if json_response.get('done', False):
                    break
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic line: {line}")

    return full_response

def display_markdown(text):
    html = markdown.markdown(text)
    print(html)  # This will print the HTML. In a web context, you'd return this instead.

# Example usage
city = "Amsterdam"
top_3_questions = generate_questions(city)
print("Raw output:")
print(top_3_questions)

print("\nFormatted as HTML:")
display_markdown(top_3_questions)

# Function to generate questions using LLaMA 3 via Ollama with a refined prompt

# def generate_questions(city):
    # Refined prompt to ask specifically for only the questions, without extra text
    #prompt = f"""
    #As a travel guide expert, generate a list of the top 10 questions that a first-time traveler might ask about visiting {city}.
    #Please provide only the questions, numbered 1 to 10, without any additional descriptions.
    

    # Use Ollama's Chat API to generate the questions
    #response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])

    #return response['message']['content']


# Function to display the questions beautifully with Markdown
def display_questions_with_markdown(city):
    # Generate the top 10 questions
    questions_text = generate_questions(city)

    # Convert the text into a Markdown-friendly format
    markdown_output = f"### Top 10 Questions for First-Time Travelers to {city}:\n\n{questions_text}"

    # Convert Markdown to HTML
    html_output = markdown.markdown(markdown_output)

    # Print the HTML output
    print(html_output)

    # Optionally, you can also print the raw Markdown if desired
    print("\nRaw Markdown:")
    print(markdown_output)

# Example usage
city = "Amsterdam"
display_questions_with_markdown(city)

"""# 5.Create Text to Speech Agent

## 5.1 ElevenLabs without Agents
"""

os.environ["ELEVEN_API_KEY"] = "MY_ELEVENLABS_API_KEY"

text_to_speak = '''Top 10 Questions for First-Time Travelers to Amsterdam:
Here are the top 10 questions that a first-time traveler might ask about visiting Amsterdam:

What is the best way to get around Amsterdam?
Is Amsterdam safe for tourists?
What are some must-see attractions in Amsterdam?
Can I drink the tap water in Amsterdam?
Are there any specific dress code or cultural norms I should be aware of?
How much money do I need to budget for food and activities?
Is Amsterdam a good place for solo travelers or couples?
What are some popular neighborhoods or areas to stay in?
Can I bring my own bike or rent one in Amsterdam?
Are there any unique or quirky experiences I should have while visiting Amsterdam?'''

tts = ElevenLabsText2SpeechTool()
print(tts.name)

speech_file = tts.run(text_to_speak)
speech_file

tts.play(speech_file)

"""## 5.2 ElevenLAbs with Agents"""

tools = load_tools(["eleven_labs_text2speech"])

agent = initialize_agent(
    tools=tools,
    llm=LLM,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
audio_file = agent.run(text_to_speak)

"""To do:
Make an AI agent which is travel agency expert and will determine all the question that traveler will have when coming to the Netherlands
What do they need to buy
where to eat
where to stay
which museum have entree with waiting list
advantages like Museum Card
which other cities are better to stay and they are close to public transport
How to arrange public transport
Travel Itinerary
Amsterdam for First Timers
"""