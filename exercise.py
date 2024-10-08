import os
import pygame
from elevenlabs import ElevenLabs, VoiceSettings
from io import BytesIO
import time
from dotenv import load_dotenv
load_dotenv()
pygame.mixer.init()

text_to_speak = '''Top 3 Questions for First-Time Travelers to Amsterdam:
Here are the top 3 questions that a first-time traveler might ask about visiting Amsterdam:

What is the best way to get around Amsterdam?
Is Amsterdam safe for tourists?
What are some must-see attractions in Amsterdam?
'''

def text_to_speech_elevenlabs(text):
    try:
        elevenclient = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
        response = elevenclient.text_to_speech.convert(
            voice_id="MF3mGyEYCl7XYWbV9V6O",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(stability=0.0, similarity_boost=1.0, style=0.0, use_speaker_boost=True),
        )

        # Write the audio data to a BytesIO object
        audio_data = BytesIO()
        for chunk in response:
            if chunk:
                audio_data.write(chunk)

        if audio_data.getbuffer().nbytes == 0:
            raise ValueError("Audio data is empty.")

        # Play the audio using pygame
        audio_data.seek(0)
        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        return "Audio played successfully"

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

    except Exception as e:
        #logging.error(f"An error occurred: {str(e)}")
        raise

result = text_to_speech_elevenlabs(text_to_speak)
print(result)