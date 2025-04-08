# YouTube Content Curator

A Flask-based web application that helps users discover and filter YouTube videos based on their preferences, extract transcripts, and find moments that match specific interests.

## Features

- Search YouTube videos by topic and user preferences
- Filter videos by minimum view count
- Extract and process video transcripts automatically
- Find and highlight preferred moments in videos based on user-defined criteria
- Summarize video content using NLP models

## Prerequisites

- Python 3.7+
- Flask
- YouTube Search Python
- YouTube Transcript API
- pandas
- FAISS for similarity search
- Transformers (for summarization)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/youtube-content-curator.git
   cd youtube-content-curator
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```
   SECRET_KEY=your-secret-key
   # Add any other API keys or configuration as needed
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

3. Enter your search topics and preferences

4. Browse the fetched videos and explore preferred moments

## Project Structure

- `app.py`: Main Flask application
- `youtube_utils.py`: Utility functions for fetching videos and transcripts
- `summarization.py`: Transcript summarization functionality
- `similarity_search.py`: Functions to identify preferred moments in videos
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JavaScript, images)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
