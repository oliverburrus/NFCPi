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
from datetime import datetime
from pathlib import Path


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



# Create the Flask app
app = Flask(__name__)

# Define the route to display the plot
@app.route("/")
def plot():
    df1 = pd.read_csv("detections.csv")
    prediction = Path('prediction.txt').read_text()
    spectrogram_path = generate_spectrogram('static/audio/sample.wav')
    table_html = df1[0:9].to_html(index=False)
    response = make_response(render_template('plot.html', table_html=table_html, prediction=prediction, spectrogram_path=spectrogram_path))
    response.headers['Refresh'] = '2'
    return response

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0')
