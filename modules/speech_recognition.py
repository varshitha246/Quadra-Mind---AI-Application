import speech_recognition as sr
from pydub import AudioSegment
import os
import logging
from utils.file_handling import convert_audio_format

# FFmpeg configuration (keep this at the top)
FFMPEG_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\Program Files\ffmpeg\bin\ffprobe.exe"
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def transcribe_audio(audio_path: str, language: str = 'en-US') -> str:
    """
    Transcribe audio file to text using Google Speech Recognition
    Handles long audio files by splitting them into 30-second chunks
    
    Args:
        audio_path: Path to the audio file
        language: Language code for speech recognition
        
    Returns:
        Transcribed text as a string
    """
    try:
        # Validate input file
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting processing for: {audio_path}")

        # Convert to WAV if needed
        if not audio_path.lower().endswith('.wav'):
            logger.info("Converting to WAV format...")
            audio_path = convert_audio_format(audio_path, 'wav')
            if not os.path.exists(audio_path):
                raise ValueError("Audio conversion failed")

        # Get duration using pydub
        audio = AudioSegment.from_wav(audio_path)
        duration_seconds = len(audio) / 1000  # Convert ms to seconds
        logger.info(f"Audio duration: {duration_seconds:.2f} seconds")

        # Initialize recognizer
        recognizer = sr.Recognizer()
        full_text = []

        # Process in 30-second chunks
        chunk_size = 30  # seconds
        for i, start_time in enumerate(range(0, int(duration_seconds), chunk_size)):
            end_time = min(start_time + chunk_size, duration_seconds)
            logger.info(f"Processing chunk {i+1}: {start_time}-{end_time}s")

            try:
                with sr.AudioFile(audio_path) as source:
                    # Seek to start time and record chunk
                    recognizer.adjust_for_ambient_noise(source)
                    audio_chunk = recognizer.record(
                        source, 
                        offset=start_time, 
                        duration=chunk_size
                    )
                    
                    text = recognizer.recognize_google(
                        audio_chunk, 
                        language=language
                    )
                    full_text.append(text)
                    
            except sr.UnknownValueError:
                logger.warning(f"Could not understand audio in chunk {i+1}")
            except sr.RequestError as e:
                logger.error(f"API error in chunk {i+1}: {e}")
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")

        if not full_text:
            raise ValueError("No speech detected in audio file")

        return ' '.join(full_text).strip()

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    try:
        result = transcribe_audio("input.mp3")
        print("Transcription Result:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")