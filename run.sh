#!/bin/bash
nohup python analyzer.py &
cd flask_app
nohup python app.py &
