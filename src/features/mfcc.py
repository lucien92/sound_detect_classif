import numpy as np
import librosa

def mfcc(y, sr):
	return 20 * np.log10(
		np.abs(
            librosa.feature.mfcc(y=y, sr=sr)
		)
	)