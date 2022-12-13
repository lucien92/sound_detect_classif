import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import config
from src.dataset.clips_dataset import get_labels
from src.features.mfcc import mfcc

audio_paths = glob.glob(f"{config['dataset_path']}/*.wav")

rms_background = np.array([])
rms_foreground = np.array([])

for audio_path in tqdm(audio_paths[:5]) :
    # Open audio file
    y, _ = librosa.load(
        path=audio_path, 
        sr=config['sr'], 
        mono=True
    )

    # Normalize
    y -= np.mean(y)
    y /= np.std(y)

    # Parse labels
    labels = get_labels(audio_path[:-3] + 'txt')
    
    # Generate background / foreground audio files
    end_label = 0
    for label in labels :
        if label['end_t'] - label['start_t'] <= 2 :
            continue

        start_label = int(label['start_t'] * config['sr'])
        if start_label > end_label :
            y_background = y[end_label : start_label]
            end_label = int(label['end_t'] * config['sr'])
            y_foreground = y[start_label : end_label]
            print(f"Label : {label['name']}, pos : {start_label} : {end_label}")

            # rms_background = np.append(rms_background, np.sqrt(np.mean(y_background**2)))
            # rms_foreground = np.append(rms_foreground, np.sqrt(np.mean(y_foreground**2)))

            # Compute power
            rms_background = np.append(rms_background, librosa.feature.rms(y=y_background, frame_length = 2 * config['sr']))
            rms_foreground = np.append(rms_foreground, librosa.feature.rms(y=y_foreground, frame_length = 2 * config['sr']))

bins = np.linspace(0, 2, 100)
plt.hist(rms_background, bins=bins, color=(0, 0, 1, 0.2))
plt.hist(rms_foreground, bins=bins, color=(1, 0, 0, 0.2))
plt.show()