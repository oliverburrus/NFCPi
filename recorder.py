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
if not os.path.exists("audio"):
    os.mkdir("audio")

x = 0
while x == 0:
 #While between sunset ans sunrise:
 print('Starting recording...')
 os.system('arecord --format=S16_LE --duration=' + str(10) + ' --rate=44100 audio/'+ str(datetime.now().strftime('%Y%m%d%H%M%S'))+'.wav')
#time.sleep(60)
#else:
 #   print("not recording...")
