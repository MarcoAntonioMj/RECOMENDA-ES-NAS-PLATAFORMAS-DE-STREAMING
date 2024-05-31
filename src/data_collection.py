import requests
import json
import os

def collect_data():
    url = "https://guilhermeonrails.github.io/api-csharp-songs/songs.json"
    response = requests.get(url)
    songs_data = response.json()

    raw_data_dir = 'data/raw/'
    os.makedirs(raw_data_dir, exist_ok=True)

    raw_data_path = os.path.join(raw_data_dir, 'songs.json')
    with open(raw_data_path, 'w') as f:
        json.dump(songs_data, f)

if __name__ == "__main__":
    collect_data()

