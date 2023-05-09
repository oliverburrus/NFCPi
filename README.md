# NFCPi
THIS PROJECT IS STILL IN DEVELOPMENT, CHECK BACK AROUND JULY 2023 FOR THE FIRST RELEASE!

NFCPi is a Raspberry Pi-based app that uses machine learning to analyze New-World Warbler (Parulidae) nocturnal flight calls in real time. The app allows users to record and analyze flight calls, view spectrograms of the recorded sounds, and track bird sightings over time. 

The process works by first recording a 20 second audio clip, which is then split up into 20, 1 second clips. These clips are then fed into a binary classifier (Convolutional Neural Network trained on the BirdVox-70k dataset) to determine if the aidio contains a nocturnal flight call. If the recording contains a nocturnal flight call, it is fed into a multiclass classifier (Convolutional Neural Network trained on the CLO43SD dataset). If the recording cannot be classified as a Warbler species, the detection is saved as passerine sp. If the model is confident in classifying a Warbler species, the recording is saved and the detection is added to the detections.csv file. 

#Dependencies 
Python 3.8+
birdnetlib
flask
librosa
matplotlib
pandas
pydub
resampy
suntime
tensorflow
