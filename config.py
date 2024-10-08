import os
from dotenv import load_dotenv

load_dotenv()

DESTINATION = "Amsterdam"
PREFERENCES = ["Museums", "Outdoor Activities"]
MAX_RESULTS = 20
MIN_VIEWS = 10000
LLM = "facebook/bart-large-cnn"
LOCAL_LLM = "llama3.2:3b-instruct-fp16"
MAX_TOKENS = 1000
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
