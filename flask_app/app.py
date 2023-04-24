from flask import Flask, render_template
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import os
from pydub import AudioSegment
import tensorflow as tf
from tensorflow.keras.models import load_model
import urllib.request
import pandas as pd
import numpy as np
import wave
import pylab
import os
from flask import Flask, render_template, make_response
import time

def generate_spectrogram(filename):
    # Load audio file
    y, sr = librosa.load(filename)

    # Generate spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), ax=ax)

    # Save the spectrogram as an image
    fig.savefig('static/images/spectrogram.png')

    # Return the path to the spectrogram image
    return 'static/images/spectrogram.png'

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def record():
    if os.path.exists("static/audio/sample.wav"):
        os.system('rm static/audio/sample.wav')
    os.system('arecord --format=S16_LE --duration=' + str(2) + ' --rate=16000 static/audio/sample.wav')

def analyze(file):
    sound_info, frame_rate = get_wav_info(file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('sample.png')

    image = tf.keras.preprocessing.image.load_img('sample.png', target_size=(256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    if not(os.path.exists('my_model.h5')):
        model_url = 'https://drive.google.com/uc?export=download&id=1cFwNVpCaMacM9fDv_2qIEOB70XkwKfKs'
        model_path = 'my_model.h5'
        # Download model file
        urllib.request.urlretrieve(model_url, model_path)
    else:
        model_path = 'my_model.h5'

    # Load model from file
    loaded_model = tf.keras.models.load_model(model_path)
    predictions = loaded_model.predict(image)
    df = pd.read_csv("https://raw.githubusercontent.com/oliverburrus/NFCPi/main/models/class_names.csv")
    # Create dataframe with class names and predictions
    df['Prediction'] = predictions[0]
    df['percentage'] = df['Prediction'].apply(lambda x: f"{x:.2%}")

    # Sort dataframe by prediction in descending order
    df = df.sort_values(by='Prediction', ascending=False)

    # Reset index
    df = df.reset_index(drop=True)
    return df

# Create the Flask app
app = Flask(__name__)

# Define the route to display the plot
@app.route("/")
def plot():
    record()
    df = analyze("static/audio/sample.wav")
    table_html = df.to_html(index=False)
    if df.Prediction[0] > .7:
        prediction = "This audio is likely of a(n) " + str(df.Species[0]) + " with a probability of " + str(df.percentage[0])
    else:
        prediction = "Not confident in my prediction"
    spectrogram_path = generate_spectrogram('static/audio/sample.wav')
    response = make_response(render_template('plot.html', table_html=table_html, prediction=prediction, spectrogram_path=spectrogram_path))
    response.headers['Refresh'] = '2'
    return response

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0')
