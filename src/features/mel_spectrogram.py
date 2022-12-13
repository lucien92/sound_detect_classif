import numpy as np
import librosa

def mel_spectrogram(y, n_fft, hop_length, n_mels):
	return 10 * np.log10(
		np.abs(
			librosa.feature.melspectrogram(y=y, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
		)
	)