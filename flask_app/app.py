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

# Create the Flask app
app = Flask(__name__)

# Define the route to display the plot
@app.route("/")
def plot():
    if os.path.exists("flask_app/bn_detections.csv"):
        df1 = pd.read_csv("flask_app/bn_detections.csv")
    else:
        new_data = ["No detections"]
        df1 = pd.DataFrame({"Species": new_data})
    prediction = Path('flask_app/prediction.txt').read_text()
    spectrogram_path = "static/images/spectrogram.png"
    table_html = df1[0:9].to_html(index=False)
    response = make_response(render_template('plot.html', table_html=table_html, prediction=prediction, spectrogram_path=spectrogram_path))
    response.headers['Refresh'] = '10'
    return response

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0')
