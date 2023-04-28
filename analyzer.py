import os
from pydub import AudioSegment
import tensorflow as tf
from tensorflow.keras.models import load_model
import urllib.request
import pandas as pd
import numpy as np
import wave
import pylab
import time
from datetime import datetime
import librosa
import matplotlib.pyplot as plt
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import date

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

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
        model_path = 'flask_app/my_model.h5'
        # Download model file
        urllib.request.urlretrieve(model_url, model_path)
    else:
        model_path = 'flask_app/my_model.h5'

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

def analyze_birdnet(file, lat, lon):
    # Load and initialize the BirdNET-Analyzer models.
    analyzer = Analyzer()

    recording = Recording(
        analyzer,
        file,
        lat=lat,
        lon=-lon,
        date=date.today(), # use date or week_48
        min_conf=0.6,
    )
    recording.analyze()
    return pd.DataFrame(recording.detections)

def generate_spectrogram(filename):
    # Load audio file
    y, sr = librosa.load(filename)

    # Generate spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), ax=ax)

    # Save the spectrogram as an image
    fig.savefig('flask_app/static/images/spectrogram.png')

    # Return the path to the spectrogram image
    #return 'static/images/spectrogram.png'

x = 0
net = "day"
aud_dir = "audio"
print("before loop")
while x == 0:
    print("before for loop")
    for filename in os.scandir(aud_dir):
        print("before if 1")
        if filename.is_file():
            name, ext = os.path.splitext(filename)
            print("before if 2")
            if ext == ".wav":
                if net == "NFC":
                    df1 = pd.DataFrame()
                    df = analyze(filename.path)
                    generate_spectrogram(filename)
                    print("1\n")
                    os.remove(filename.path)
                    print("2\n")
                    if df.Prediction[0] > .7:
                        prediction = "This audio is likely of a(n) " + str(df.Species[0]) + " with a probability of " + str(df.percentage[0])
                        new_data = pd.DataFrame({'Species': [df.Species[0]], "Probability": [df.percentage[0]], "DT": [datetime.now()]})
                        if os.path.exists("flask_app/detections.csv"):
                            df1 = pd.read_csv("flask_app/detections.csv")
                            df1 = pd.concat([df1, new_data], ignore_index=True)
                        else:
                            df2 = pd.DataFrame(columns=['Species', 'Probability', 'DT'])
                            df1 = pd.concat([df2, new_data], ignore_index=True)
                        df1.to_csv("flask_app/detections.csv", index=False)
                    else:
                        print("3\n")
                        prediction = "Not confident in my prediction"

                    text_file = open("flask_app/prediction.txt", "w") 
                    print("4\n")

                    # Writing the file 
                    text_file.write(prediction) 
                    text_file.close()
                elif net == "day":
                    df1 = pd.DataFrame()
                    df = analyze_birdnet(filename.path, 43.9, -90.0)
                    generate_spectrogram(filename)
                    print("1\n")
                    os.remove(filename.path)
                    if my_list:
                        prediction = "Found a match!"
                        if os.path.exists("flask_app/bn_detections.csv"):
                            df1 = pd.read_csv("flask_app/bn_detections.csv")
                            df1 = pd.concat([df1, df], ignore_index=True)
                        else:
                            df1 = pd.DataFrame(columns=['common_name', 'confidence', 'end_time', 'scientific_name', 'start_time'])
                            df1 = pd.concat([df1, df], ignore_index=True)
                        df1.to_csv("flask_app/bn_detections.csv", index=False)
                    else:
                        print("3\n")
                        prediction = "Not confident in my prediction"
                    text_file = open("flask_app/prediction.txt", "w") 
                    print("4\n")

                    # Writing the file 
                    text_file.write(prediction) 
                    text_file.close()
            else:
                print(f"Ignoring file {filename}, not a .wav file")
                continue
    time.sleep(1)
            
# Print dataframe
#TODO: function to save file and remove old file
#time = pd.to_datetime(df['Time (hh:mm:ss)'], format='%H:%M:%S.%f', utc = True)


#audio = AudioSegment.from_file(os.getcwd() + '\\sample.wav', 'wav')

#for i in range(0, df.shape[0]):
 #   ms = time[i].hour*3600000+time[i].minute*60000+time[i].second*1000
  #  bird = audio[ms-1500:ms+1500]
   # bird.export(str(ms)+str(df['Species (4-letter code)'][i])+'.wav', format = 'wav')
    
#os.system(r'rm ' + os.getcwd() + '/sample.wav')
