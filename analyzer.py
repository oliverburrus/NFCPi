import birdvoxdetect as bvd   
import os
from pydub import AudioSegment

df = bvd.process_file(os.getcwd() + '/mowaq941.wav')

import pandas as pd
time = pd.to_datetime(df['Time (hh:mm:ss)'], format='%H:%M:%S.%f', utc = True)


audio = AudioSegment.from_file(os.getcwd() + '\\sample.wav', 'wav')

for i in range(0, df.shape[0]):
    ms = time[i].hour*3600000+time[i].minute*60000+time[i].second*1000
    bird = audio[ms-1500:ms+1500]
    bird.export(str(ms)+str(df['Species (4-letter code)'][i])+'.wav', format = 'wav')
    
os.system(r'rm ' + os.getcwd() + '/sample.wav')
