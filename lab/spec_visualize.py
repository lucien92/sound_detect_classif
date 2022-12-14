import glob
import numpy as np
import librosa
from scipy import signal
import matplotlib.pyplot as plt

from src.config import config
from src.dataset.clips_dataset import get_labels
from src.utils.visualize import visualize

audio_paths = glob.glob(f"{config['dataset_path']}/*.wav")
audio_path = audio_paths[0]
y, _ = librosa.load(
    path=audio_path, 
    sr=config['sr'], 
    mono=True
)

# f, t, Y = signal.stft(y, config['sr'], nperseg=config['n_fft'], noverlap=config['n_fft']-config['hop_length'])
# S = 20 * np.log(np.abs(Y))
# labels = get_labels(audio_path[:-3] + 'txt')
# visualize(S, labels, t, f)

t = np.arange(len(y)) / config['sr']
labels = get_labels(audio_path[:-3] + 'txt')
visualize(y, labels, t)