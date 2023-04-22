from flask import Flask, render_template
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

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
    spectrogram_path = generate_spectrogram('static/audio/sample.wav')
    return render_template('plot.html', spectrogram_path=spectrogram_path)

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0')
