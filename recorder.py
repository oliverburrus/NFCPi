import os
import datetime
from suntime import Sun, SunTimeException

latitude = 51.21
longitude = 21.01

sun = Sun(latitude, longitude)

# Get today's sunrise and sunset in UTC
today_sr = sun.get_sunrise_time()
today_ss = sun.get_sunset_time()

# Given timestamp in string
time_str_sr = today_sr
date_format_str = '%d/%m/%Y %H:%M'
# create datetime object from timestamp string
given_time1 = datetime.strptime(time_str_sr, date_format_str)

n = 2
# Add 2 hours to datetime object
final_time_sr = given_time1 + timedelta(hours=n)

# Given timestamp in string
time_str_ss = today_ss
# create datetime object from timestamp string
given_time = datetime.strptime(time_str_ss, date_format_str)

# Subtract 2 hours to datetime object
final_time_ss = given_time - timedelta(hours=n)

duration = final_time_ss-final_time_sr
# current datetime
now = datetime.datetime.now()

current_date = now.date()

current_time = now.time()

print(duration)

if now.t
os.system('arecord --format=S16_LE --duration=' + duration + ' --rate=16000 ' + 'sample.wav')
