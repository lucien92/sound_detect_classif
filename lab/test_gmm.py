import librosa
import numpy as np
import sklearn.mixture
import random

from src.config import config
from src.dataset.clips_dataset import clips_dataset
from src.features.mfcc import mfcc

# Load dataset
dataset = clips_dataset()
random.Random().shuffle(dataset)

# Process data
X = {}
X["Hyla meridionalis"] = np.zeros((1, 20))
X["Pelodytes punctatus"] = np.zeros((1, 20))
for data in dataset[:400]:
    if data['name'] == "Hyla meridionalis" or data['name'] == "Pelodytes punctatus":
        # print(f"Processing : {data['path']}")
        y, sr = librosa.load(
			path=data['path'], 
			sr=config['sr'], 
			mono=True,
			offset=data['start_t'],
			duration=config['duration']
		)
        features = mfcc(y, sr).T
        X[data['name']] = np.append(X[data['name']], features, axis=0)

# Train
gmm = {}
gmm["Hyla meridionalis"] = sklearn.mixture.GaussianMixture(n_components=16)
gmm["Hyla meridionalis"].fit(X["Hyla meridionalis"])
gmm["Pelodytes punctatus"] = sklearn.mixture.GaussianMixture(n_components=16)
gmm["Pelodytes punctatus"].fit(X["Pelodytes punctatus"])

# Predict
good = 0
total = 0
for data in dataset[400:600]:
    if data['name'] == "Hyla meridionalis" or data['name'] == "Pelodytes punctatus":
        print(f"Processing : {data['path']}")
        print(f"Type : {data['name']}")
        y, sr = librosa.load(
			path=data['path'], 
			sr=config['sr'], 
			mono=True,
			offset=data['start_t'],
			duration=config['duration']
		)
        features = mfcc(y, sr).T
        
        scores = []
        for gmm_name, gmm_item in gmm.items():
            scores.append((gmm_name, gmm_item.score(features)))
        res = sorted(scores, key=lambda x: x[1], reverse=True)

        if(res[0][0] == data['name']):
            good += 1
        
        total += 1

print(good / total)