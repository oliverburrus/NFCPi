#!/bin/sh

pip install birdvoxdetect
pip install pydub
pip install datetime
pip install suntime
git clone https://github.com/oliverburrus/NFCPi.git
cd NFCPi
chmod +x recorder.py
chmod +x analyzer.py
nohup python recorder.py &
