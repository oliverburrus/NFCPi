from flask import Flask, render_template, url_for
import os
import pandas as pd
from pathlib import Path

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
    audio = [url_for('static', filename=f'audio/{f}') for f in audio_files]

    # render the audio.html template with the audio file URLs
    return render_template('audio_list.html', audio=audio)
    
@app.route("/stats")
def stats():
    if os.path.exists("bn_detections.csv"):
        df = pd.read_csv("bn_detections.csv")
    else:
        new_data = {"common_name": ["No detections"], "confidence": [0], "end_time": [0], "scientific_name": ["None"], "start_time": [0], "timestamp": ["None"]}
        df = pd.DataFrame(new_data)

    # Create a bar plot of the top 10 common names
    top_common_names = df['common_name'].value_counts().head(10)
    plt.bar(top_common_names.index, top_common_names.values)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Common Name')
    plt.ylabel('Number of Detections')
    plt.tight_layout()
    common_names_path = "static/images/common_names.png"
    plt.savefig(common_names_path)
    plt.clf()

    # Create a histogram of the confidence values
    plt.hist(df['confidence'], bins=20)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.tight_layout()
    confidence_path = "static/images/confidence.png"
    plt.savefig(confidence_path)
    plt.clf()

    # Create a scatter plot of start time vs. confidence
    plt.scatter(df['start_time'], df['confidence'])
    plt.xlabel('Start Time')
    plt.ylabel('Confidence')
    plt.tight_layout()
    start_time_path = "static/images/start_time.png"
    plt.savefig(start_time_path)
    plt.clf()

    # Pass the paths to the generated images and summary stats to the stats.html template
    return render_template('stats.html', common_names_path=common_names_path, confidence_path=confidence_path, start_time_path=start_time_path, num_detections=len(df), num_species=len(df['scientific_name'].unique()))


# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0')
