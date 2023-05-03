#!/bin/bash
python recorder.py &
python analyzer.py &
python flask_app/app.py
