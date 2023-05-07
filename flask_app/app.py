from flask import Flask, render_template
import os

# Create the Flask app
app = Flask(__name__)

# Define the route to display the plot
@app.route("/")
def plot():
    if os.path.exists("bn_detections.csv"):
        df1 = pd.read_csv("bn_detections.csv")
    else:
        new_data = ["No detections"]
        df1 = pd.DataFrame({"Species": new_data})
    prediction = Path('prediction.txt').read_text()
    spectrogram_path = "static/images/spectrogram.png"
    last_detect_path = "static/images/last_detect.png"
    table_html = df1[0:9].to_html(index=False)
    return render_template('plot.html', table_html=table_html, prediction=prediction, spectrogram_path=spectrogram_path, last_detect_path=last_detect_path)

# Define the route to display audio recordings
@app.route("/audio")
def audio():
    # set directory path
    audio_dir = os.path.join(app.static_folder, 'audio')

    # get a list of audio files in the directory
    audio_files = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f)) and f.endswith('.wav')]

    # create a list of audio file URLs
    audio_urls = [url_for('static', filename=f'audio/{f}') for f in audio_files]

    # render the audio.html template with the audio file URLs
    return render_template('audio.html', audio_urls=audio_urls)

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0')
