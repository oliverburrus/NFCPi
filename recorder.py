import os
from datetime import datetime
from datetime import timedelta
from suntime import Sun, SunTimeException
import time

latitude = 42.11
longitude = -88.27

sun = Sun(latitude, longitude)

# Get today's sunrise and sunset in UTC
today_sr = sun.get_sunrise_time()-timedelta(hours = 2)
today_ss = sun.get_sunset_time()+timedelta(hours = 2)


duration = today_sr-today_ss
# current datetime
now = datetime.now()

current_date = now.date()

current_time = now.time()

tsecs = round(duration.total_seconds())

now = datetime.utcnow()
if now.hour==today_ss.hour and now.minute==today_ss.minute:
    print('Starting recording...')
    os.system('arecord --format=S16_LE --duration=' + str(tsecs) + ' --rate=16000 sample.wav')
    time.sleep(60)
else:
    print("not recording...")
