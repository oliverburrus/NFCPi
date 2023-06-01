import sounddevice as sd
import numpy as np
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
import os
import urllib

# Set up audio parameters
sample_rate = 16000  # Sample rate in Hz
duration = 0.5  # Duration of each audio frame in seconds
n_fft = 2048  # Number of FFT points

def get_models():
    print("Getting models ready...")
    if not(os.path.exists('flask_app/binary.h5')):
        model_url = 'https://drive.google.com/uc?export=download&id=14igHOLLg74WiM-eTHPVA9sKs_hAmiuVr'
        model_path = 'flask_app/binary.h5'
        # Download model file
        urllib.request.urlretrieve(model_url, model_path)
    else:
        model_path = 'flask_app/binary.h5'
    # Load model from file
    binary_model = tf.keras.models.load_model(model_path)

    return binary_model

# Load the trained CNN model
model = get_models()

def save_spectrogram(spectrogram):
    # Convert spectrogram to a plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax)

    # Save the plot as a PNG image
    plt.savefig('sample.png')
    plt.close(fig)

# Define a variable to store the history of audio frames and spectrograms
audio_history = []
spectrogram_history = []
total_samples = int(sample_rate * 10)  # Total number of samples to keep in history

# Define the audio callback function
def audio_callback(indata, frames, time, status):
    # Preprocess the audio data
    audio = indata[:, 0]  # Consider only the first channel if using stereo input
    audio = audio / np.max(np.abs(audio))  # Normalize the audio amplitudes

    # Compute the spectrogram
    nperseg = min(len(audio), int(sample_rate * duration))
    frequencies, times, spectrogram = signal.spectrogram(audio, fs=sample_rate, scaling='spectrum', mode='magnitude', nfft=n_fft, window=('tukey', 0.25), nperseg=nperseg)
    spectrogram = 10 * np.log10(spectrogram)  # Convert to dB scale

    # Add current audio and spectrogram to the history
    audio_history.append(audio)
    spectrogram_history.append(spectrogram)

    # Remove old audio and spectrogram from the history if necessary
    if len(audio_history) > total_samples:
        audio_history.pop(0)
        spectrogram_history.pop(0)

    # Perform inference with the CNN model on the current spectrogram
    if len(audio_history) >= nperseg:
        audio_concat = np.concatenate(audio_history)
        spectrogram_concat = np.concatenate(spectrogram_history, axis=1)
        audio_history.clear()
        spectrogram_history.clear()

        # Take the most recent segment for inference
        start_idx = len(audio_concat) - nperseg
        audio_segment = audio_concat[start_idx:]
        spectrogram_segment = spectrogram_concat[:, -nperseg:]

        # Process the audio segment and spectrogram
        current_spectrogram = np.expand_dims(spectrogram_segment, axis=-1)  # Add an extra dimension to match the input shape of the model
        current_spectrogram = np.repeat(current_spectrogram, 3, axis=-1)  # Replicate the single-channel spectrogram along the channel axis
        current_spectrogram = np.expand_dims(current_spectrogram, axis=0)  # Add a batch dimension
        resized_spectrogram = tf.image.resize(current_spectrogram, size=(256, 256))  # Resize spectrogram to desired dimensions
        normalized_spectrogram = resized_spectrogram / 255.0  # Normalize the spectrogram

        predictions = model.predict(normalized_spectrogram)
        # Process the predictions or perform further analysis based on your specific task

        save_spectrogram(spectrogram_segment)  # Replace `save_spectrogram` with your desired saving function


# Start the audio stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate)
stream.start()

# Keep the script running until interrupted
while True:
    pass

# Stop the audio stream
stream.stop()
stream.close()

