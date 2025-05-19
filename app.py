from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename # This should now be correctly imported from werkzeug.utils
from werkzeug.utils import secure_filename
from flask_uploads import UploadSet, configure_uploads
IMAGES = ('jpg', 'jpeg', 'png', 'gif')
AUDIO = ('wav', 'mp3', 'ogg')
import os
from modules.summarization import summarize_text
from modules.speech_recognition import transcribe_audio
from modules.neural_style_transfer import perform_style_transfer
from modules.text_generation import generate_text
# Note: save_uploaded_file from utils.file_handling seems unused in the provided routes, but keeping the import
from utils.file_handling import allowed_file
import uuid

app = Flask(__name__)
# IMPORTANT: Change 'your-secret-key-here' to a strong, unique, random value in production!
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# --- Configuration for flask_uploads destinations ---
# These specify the subdirectories within UPLOAD_FOLDER for each UploadSet
app.config['UPLOADED_AUDIOS_DEST'] = os.path.join(app.config['UPLOAD_FOLDER'], 'audio')
app.config['UPLOADED_IMAGES_DEST'] = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
# --- End flask_uploads configuration ---


# Configure file uploads with the Flask application
# This step checks that the UPLOADED_..._DEST configurations are set
audios = UploadSet('audios', AUDIO)
images = UploadSet('images', IMAGES)
configure_uploads(app, (audios, images))


@app.route('/')
def index():
    # Renders the index.html template, which should extend base.html
    return render_template('index.html')

@app.route('/summarization', methods=['GET', 'POST'])
def summarization():
    if request.method == 'POST':
        text = request.form.get('text')
        # Basic validation for empty input text
        if not text or not text.strip():
             return render_template('summarization.html', error="Please enter text to summarize.")

        try:
            summary = summarize_text(text)
            return render_template('summarization.html', original_text=text, summary=summary)
        except Exception as e:
            # Catch any errors during summarization
            print(f"Error during summarization: {e}") # Log the error to the terminal
            return render_template('summarization.html', original_text=text, error=f"Error during summarization: {e}")

    # Render the empty summarization form for GET requests
    return render_template('summarization.html')

@app.route('/speech-recognition', methods=['GET', 'POST'])
def speech_recognition():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return render_template('speech.html', error="No audio file selected.")

        audio_file = request.files['audio']

        if audio_file.filename == '':
            return render_template('speech.html', error="No audio file selected.")

        if audio_file and allowed_file(audio_file.filename, {'wav', 'mp3', 'ogg'}):
            try:
                # Generate a secure filename
                filename = secure_filename(audio_file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                filepath = os.path.join(app.config['UPLOADED_AUDIOS_DEST'], unique_filename)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save the file
                audio_file.save(filepath)
                app.logger.info(f"Audio file saved to: {filepath}")

                try:
                    transcript = transcribe_audio(filepath)
                    return render_template('speech.html', 
                                         transcript=transcript,
                                         audio_file=unique_filename)
                except ValueError as e:
                    return render_template('speech.html', 
                                         error=str(e))
                except Exception as e:
                    app.logger.error(f"Transcription error: {str(e)}")
                    return render_template('speech.html', 
                                         error=f"Transcription failed: {str(e)}")
                finally:
                    # Clean up the file after processing
                    if os.path.exists(filepath):
                        os.remove(filepath)
            
            except Exception as e:
                app.logger.error(f"Error processing audio file: {str(e)}")
                return render_template('speech.html', 
                                     error=f"Error processing audio file: {str(e)}")
        
        else:
            return render_template('speech.html', 
                               error="Invalid file type. Please upload WAV, MP3, or OGG audio files.")

    return render_template('speech.html')

@app.route('/style-transfer', methods=['GET', 'POST'])
def style_transfer():
    if request.method == 'POST':
        # Check if both content and style image files are in the request
        if 'content' not in request.files or 'style' not in request.files:
            return render_template('style_transfer.html', error="Missing content or style image file.")

        content_file = request.files['content']
        style_file = request.files['style']

        # Check if filenames are empty (e.g., if user submitted without selecting files)
        if content_file.filename == '' or style_file.filename == '':
            return render_template('style_transfer.html', error="Content or style image filename is empty.")

        # Validate file types for both uploaded images
        allowed_image_extensions = {'jpg', 'jpeg', 'png'}
        if (content_file and allowed_file(content_file.filename, allowed_image_extensions)) and \
           (style_file and allowed_file(style_file.filename, allowed_image_extensions)):

            # Generate unique filenames using uuid to prevent naming conflicts
            content_filename = f"content_{uuid.uuid4().hex}.{content_file.filename.split('.')[-1].lower()}"
            style_filename = f"style_{uuid.uuid4().hex}.{style_file.filename.split('.')[-1].lower()}"

            # Define file paths using the configured image upload destination
            content_path = os.path.join(app.config['UPLOADED_IMAGES_DEST'], content_filename)
            style_path = os.path.join(app.config['UPLOADED_IMAGES_DEST'], style_filename)

            # Define the intended output path for the styled image
            # Output is saved in the 'outputs' subdirectory of the main upload folder
            output_filename = f"output_{uuid.uuid4().hex}.jpg" # Output as JPG is common
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'outputs', output_filename)

            try:
                # Save the uploaded content and style files to the server
                content_file.save(content_path)
                style_file.save(style_path)

                # Perform style transfer using the module function
                # The function is expected to save the result to output_path
                # It returns a tuple: (success_status, message_or_path)
                success, result_info = perform_style_transfer(content_path, style_path, output_path)

                # Clean up uploaded files immediately if not needed after processing (optional)
                # os.remove(content_path)
                # os.remove(style_path)

                if success:
                    # If style transfer was successful
                    # Pass paths relative to 'static' for the template to display images
                    # These paths are correct for use with url_for('static', filename=...)
                    output_relative_path = f'uploads/outputs/{output_filename}'
                    content_relative_path = f'uploads/images/{content_filename}'
                    style_relative_path = f'uploads/images/{style_filename}'

                    # Render the style_transfer.html template with input and output images
                    return render_template('style_transfer.html',
                                           content_image=content_relative_path,
                                           style_image=style_relative_path,
                                           output_image=output_relative_path) # Pass the correct relative path


                else:
                    # If style transfer failed (error reported by perform_style_transfer)
                    error_message = result_info # The function returned the error message
                    print(f"Style transfer failed: {error_message}") # Log the error
                    return render_template('style_transfer.html', error=f"Style transfer failed: {error_message}")


            except Exception as e:
                 # Catch any unexpected errors during file saving or function call
                 print(f"An unexpected error occurred during style transfer process: {e}") # Log the error
                 # Clean up potentially saved files if an error occurs (optional)
                 if os.path.exists(content_path): os.remove(content_path)
                 if os.path.exists(style_path): os.remove(style_path)
                 return render_template('style_transfer.html', error=f"An unexpected error occurred: {e}")
        else:
            # If file type is not allowed for either content or style image
            return render_template('style_transfer.html', error=f"Invalid file type. Please upload {', '.join(allowed_image_extensions)} images.")

    # Render the empty style transfer form for GET requests
    return render_template('style_transfer.html')


@app.route('/text-generation', methods=['GET', 'POST'])
def text_generation():
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        # Get length, default to 100 if not provided
        length_str = request.form.get('length', '100')

        # Basic validation for empty prompt
        if not prompt or not prompt.strip():
             return render_template('generation.html', error="Please enter a prompt for text generation.")

        # Validate and convert length to integer
        try:
            length = int(length_str)
            # Ensure length is a positive number
            if length <= 0:
                 return render_template('generation.html', error="Length must be a positive number.")
            # Optional: Set a maximum length to prevent excessive computation
            max_length = 500 # Example max length
            if length > max_length:
                 return render_template('generation.html', error=f"Length exceeds maximum allowed ({max_length}).")

        except ValueError:
            # Handle case where length is not a valid integer
            return render_template('generation.html', error="Invalid length provided. Please enter a number.")

        try:
            # Generate text using the module function
            generated_text = generate_text(prompt, length)
            # Render the generation.html template with the result
            return render_template('generation.html', prompt=prompt, generated_text=generated_text)
        except Exception as e:
            # Catch any errors during text generation
            print(f"Error during text generation: {e}") # Log the error
            return render_template('generation.html', prompt=prompt, error=f"Error during text generation: {e}")


    # Render the empty text generation form for GET requests
    return render_template('generation.html')


# This block ensures the upload directories are created when the script is run directly
if __name__ == '__main__':
    # Construct the full paths for the upload directories
    upload_folder = app.config['UPLOAD_FOLDER']
    audio_upload_dir = os.path.join(upload_folder, 'audio')
    images_upload_dir = os.path.join(upload_folder, 'images')
    outputs_dir = os.path.join(upload_folder, 'outputs')

    # Create the directories if they do not already exist
    os.makedirs(upload_folder, exist_ok=True) # Ensure the base upload folder exists
    os.makedirs(audio_upload_dir, exist_ok=True)
    os.makedirs(images_upload_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True) # Ensure the outputs folder also exists


    # Run the Flask development server
    # debug=True provides helpful error pages and reloads the server on code changes
    app.run(debug=True)

