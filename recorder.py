import os
from datetime import datetime
from datetime import timedelta
from suntime import Sun, SunTimeException
import schedule
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

def record():
  os.system('arecord --format=S16_LE --duration=' + str(tsecs) + ' --rate=16000 ' + 'sample.wav')

schedule.every().day.at(today_ss.time()).do(record())

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute

