import birdvoxdetect as bvd   
import os
from pydub import AudioSegment
import tensorflow as tf
from tensorflow.keras.models import load_model
import urllib.request
import pandas as pd
import numpy as np


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

x = 0
aud_dir = "audio"
while x == 0:
    for filename in os.scandir(aud_dir):
        if filename.is_file():
            df1 = pd.DataFrame()
            record()
            df = analyze("static/audio/sample.wav")
            if df.Prediction[0] > .7:
                prediction = "This audio is likely of a(n) " + str(df.Species[0]) + " with a probability of " + str(df.percentage[0])
                new_data = pd.DataFrame({'Species': [df.Species[0]], "Probability": [df.percentage[0]], "DT": [datetime.now()]})
                if os.path.exists("flask_app/detections.csv"):
                    df1 = pd.read_csv("flask_app/detections.csv")
                    df1 = pd.concat([df1, new_data], ignore_index=True)
                else:
                    df1 = df1.append(new_data, ignore_index=True)
                df1.to_csv("flask_app/detections.csv", index=False)
            else:
                prediction = "Not confident in my prediction"

            text_file = open("flask_app/prediction.txt", "w") 

            # Writing the file 
            text_file.write(prediction) 
            text_file.close() 
# Print dataframe
#TODO: function to save file and remove old file
#time = pd.to_datetime(df['Time (hh:mm:ss)'], format='%H:%M:%S.%f', utc = True)


#audio = AudioSegment.from_file(os.getcwd() + '\\sample.wav', 'wav')

#for i in range(0, df.shape[0]):
 #   ms = time[i].hour*3600000+time[i].minute*60000+time[i].second*1000
  #  bird = audio[ms-1500:ms+1500]
   # bird.export(str(ms)+str(df['Species (4-letter code)'][i])+'.wav', format = 'wav')
    
#os.system(r'rm ' + os.getcwd() + '/sample.wav')
