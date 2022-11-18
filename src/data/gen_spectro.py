import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import csv

def spectrogram(y, n_fft, hop_length):
	return 20 * np.log10(np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)))

def melgram(y, n_fft, hop_length, n_mels):
	return 20 * np.log10(
		np.abs(
			librosa.feature.melspectrogram(y, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
		)
	)

base_path = "/home/david/Escriptori/Feines/sound_detect_classif/src/data/Spectrograms/"

try:
    os.mkdir(base_path)
except:
    pass


with open("/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/classic_data.csv", "r") as f:
    lines = f.readlines()
    for line in lines:
        
        print(line)
        wav_doc = line.split(",")[0]
        waveform, _ = librosa.load(
            path=wav_doc, 
            sr=256*64
        ) 

        feature = spectrogram(waveform, 512, 256) #nfft et hop_length, nfft est la taille de la fenetre, hop_length est le pas entre deux fenetres


        #on veut transofrmer cet array en image

        plt.figure(figsize=(10, 4))
        #on veut enlever les marges blanches
        plt.axis('off')
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        sr = librosa.get_samplerate(wav_doc) 
        librosa.display.specshow(feature, sr=sr, x_axis='time', y_axis='linear')
        plt.savefig(f"{base_path}{wav_doc.rsplit('/', 1)[1][:-4]}.png", bbox_inches=None, pad_inches=0)
        #plt.show()
        plt.close()

        #Il y a différentes fréquence d'échantillonage(sampling rate) (cela pour capter différents bruits). 
        #Le sampling rate vaut deux fois la valeur maximale captée.
        #Mais comme le sampling rate varie selon les enregistrements, on va normaliser les spectogrammes
        # sr = librosa.get_samplerate(wav_doc) 
        # print(sr)
        # librosa.display.specshow(feature, sr=sr, fmax = 48000) #sr est la frequence d'echantillonagemax multipliée par 2
        # plt.savefig(base_path + f"{wav_doc[:-4]}.png")
        # plt.show()
        


