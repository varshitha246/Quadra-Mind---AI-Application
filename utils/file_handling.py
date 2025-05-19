from pydub import AudioSegment
import os
from werkzeug.utils import secure_filename
import logging

logger = logging.getLogger(__name__)

def allowed_file(filename, allowed_extensions):
    """
    Check if file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def convert_audio_format(input_path, target_format='wav'):
    """
    Convert audio file to target format using pydub/ffmpeg
    Returns path to converted file
    """
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Create output path
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}.{target_format.lower()}"
        
        # Convert using pydub
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format=target_format)
        
        logger.info(f"Converted {input_path} to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting audio file: {str(e)}")
        raise ValueError(f"Could not convert audio file: {str(e)}")

def save_uploaded_file(file, upload_folder, allowed_extensions):
    """
    Save uploaded file with secure filename
    Returns saved file path if successful, None otherwise
    """
    if file and allowed_file(file.filename, allowed_extensions):
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        os.makedirs(upload_folder, exist_ok=True)
        file.save(filepath)
        return filepath
    return None