import birdvoxdetect as bvd   
import os
from pydub import AudioSegment

df = bvd.process_file(os.getcwd() + 'sample.wav')
df['Time (hh:mm:ss)'] = pd.to_datetime(df['Time (hh:mm:ss)'], format='%H:%M:%S')
ms = df['Time (hh:mm:ss)'].hour*3600000+df['Time (hh:mm:ss)'].minute*60000+df['Time (hh:mm:ss)'].second*1000
audio = AudioSegment.from_file(os.getcwd() + 'sample.wav', 'wav')

for i in 0:df.shape[0]:
    bird = audio[ms[i]-1500:ms[i]+1500]
    bird.export(ms+df['Species (4-letter code)'][i]+'.wav', format = 'wav')
    
os.system('rm sample.wav')
