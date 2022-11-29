import keras
import math
import numpy as np
import librosa
import pandas as pd

from ..config import config
from ..features.mel_spectrogram import mel_spectrogram

class Generator(keras.utils.Sequence):

	def __init__(self, dataset, batch_size, labels) -> None:
		super().__init__()
		self.dataset 	= dataset
		self.batch_size = batch_size
		self.input_shape = (
			config['n_mels'], 
			config['duration'] * config['sr'] // config['hop_length'] + 1, 
			1
		)
		print('IS :', self.input_shape)

		# self.df = pd.read_excel('data/raw/Meta_data_acoustique.xlsx')
		# self.species_names = self.df['Espèce cible'].unique()
		# self.n_classes = len(self.species_names)
		self.labels = labels
		self.n_classes = len(self.labels)

	def __len__(self) -> int:
		return math.floor(len(self.dataset) / self.batch_size)

	def __getitem__(self, index):
		indexes = range(index * self.batch_size, (index + 1) * self.batch_size)

		x = np.empty((self.batch_size, *self.input_shape))
		y = np.empty((self.batch_size))
		for i in range(self.batch_size):
			x[i, ]  = self._extract_feature(self.dataset[indexes[i]])
			y[i]    = self._extract_label(self.dataset[indexes[i]]['name'])

		y = keras.utils.to_categorical(y, num_classes=self.n_classes)
		return x, y

	def _extract_feature(self, data):
		y, _ = librosa.load(
			path=data['path'], 
			sr=config['sr'], 
			mono=True,
			offset=data['start_t'],
			duration=config['duration']
		)
		feature = mel_spectrogram(y, config['n_fft'], config['hop_length'], config['n_mels'])
		feature = np.pad(feature, ((0, 0), (0, self.input_shape[1] - feature.shape[1])), 'constant', constant_values=0)
		feature = np.expand_dims(feature, axis=-1)
		return feature

	def _extract_label(self, label):
		where = np.where(self.labels == label)
		id = where[0][0] if len(where[0]) != 0 else 0
		return id