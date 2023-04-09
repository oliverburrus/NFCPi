import requests

# Create the observation
observation_data = {
    "observation": {
        "taxon_id": 1,
        "latitude": 37.123,
        "longitude": -122.456,
        "observed_on_string": "2023-04-08",
        "time_zone": "Pacific Time (US & Canada)",
        "description": "A description of the observation",
    }
}
response = requests.post("https://api.inaturalist.org/v1/observations", json=observation_data)
observation_id = response.json()["id"]

# Upload the audio file
file_data = {"file": ("audio.mp3", open("audio.mp3", "rb"))}
response = requests.post(f"https://api.inaturalist.org/v1/observation_sounds?observation_id={observation_id}", files=file_data)
sound_id = response.json()["id"]

# Associate the audio file with the observation
observation_data = {
    "observation": {
        "sounds_attributes": [{"id": sound_id}]
    }
}
response = requests.put(f"https://api.inaturalist.org/v1/observations/{observation_id}", json=observation_data)
