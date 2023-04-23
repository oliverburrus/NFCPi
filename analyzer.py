import birdvoxdetect as bvd   
import os
from pydub import AudioSegment
import tensorflow as tf
from tensorflow.keras.models import load_model
import urllib.request
import pandas as pd
import numpy as np

min_probability = .8

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

sound_info, frame_rate = get_wav_info('sample.wav')
pylab.specgram(sound_info, Fs=frame_rate)
pylab.savefig('sample.png')

image = tf.keras.preprocessing.image.load_img('sample.png', target_size=(256, 256))
image = tf.keras.preprocessing.image.img_to_array(image)
image /= 255.0
image = np.expand_dims(image, axis=0)
if !(os.path.exists('my_model.h5')):
    model_url = 'https://drive.google.com/uc?export=download&id=1cFwNVpCaMacM9fDv_2qIEOB70XkwKfKs'
    model_path = 'my_model.h5'
    # Download model file
    urllib.request.urlretrieve(model_url, model_path)
else:
    model_path = 'my_model.h5'

# Load model from file
loaded_model = load_model(model_path)
predictions = model.predict(image)
predicted_class = np.argmax(predictions)

# Create dataframe with class names and predictions
df = pd.DataFrame({'Class': class_names, 'Prediction': predictions[0]})
df['percentage'] = df['Prediction'].apply(lambda x: f"{x:.2%}")


# Sort dataframe by prediction in descending order
df = df.sort_values(by='Prediction', ascending=False)

# Reset index
df = df.reset_index(drop=True)

# Print dataframe
print(df[0:5])

time = pd.to_datetime(df['Time (hh:mm:ss)'], format='%H:%M:%S.%f', utc = True)


audio = AudioSegment.from_file(os.getcwd() + '\\sample.wav', 'wav')

for i in range(0, df.shape[0]):
    ms = time[i].hour*3600000+time[i].minute*60000+time[i].second*1000
    bird = audio[ms-1500:ms+1500]
    bird.export(str(ms)+str(df['Species (4-letter code)'][i])+'.wav', format = 'wav')
    
os.system(r'rm ' + os.getcwd() + '/sample.wav')
