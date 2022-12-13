import glob
import numpy as np
import librosa
from scipy import signal
import matplotlib.pyplot as plt

from src.config import config
from src.dataset.clips_dataset import get_labels

def visualize(feature, t, f, labels) :
    fig, ax = plt.subplots()
    plt.title(labels[0]['path'])
    plt.imshow(feature, aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')

    for label in labels :
        rect = plt.Rectangle(
            (label['start_t'], label['start_f']), label['end_t'] - label['start_t'], label['end_f'] - label['start_f'],
            facecolor=(1,0,0,.2),
            edgecolor='red',
            lw=2, 
            # transform=fig.transFigure, figure=fig
        )
        ax.add_patch(rect)
        ax.text(
            0.5*(label['start_t']+label['end_t']), 0.5*(label['start_f']+label['end_f']), label['name'],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10, color='black',
            # transform=fig.transFigure
        )

    plt.show()

audio_paths = glob.glob(f"{config['dataset_path']}/*.wav")
audio_path = audio_paths[0]
y, _ = librosa.load(
    path=audio_path, 
    sr=config['sr'], 
    mono=True
)
f, t, Y = signal.stft(y, config['sr'], nperseg=config['n_fft'], noverlap=config['n_fft']-config['hop_length'])
S = 20 * np.log(np.abs(Y))
labels = get_labels(audio_path[:-3] + 'txt')
visualize(S, t, f, labels)