import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt

from src.config import config
from src.dataset.clips_dataset import get_labels
from src.features.mfcc import mfcc

audio_paths = glob.glob(f"{config['dataset_path']}/*.wav")

y, _ = librosa.load(
    path=audio_paths[8], 
    sr=config['sr'], 
    mono=True
)

rms = librosa.feature.rms(y=y, frame_length = 3 * config['sr'], hop_length = 1 * config['sr'])[0, :]

plt.subplot(2, 1, 1)
plt.plot(y)
plt.subplot(2, 1, 2)
times = np.linspace(0, len(y), num=len(rms), endpoint=False)
plt.plot(times, rms)
plt.show()