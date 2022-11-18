import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import csv
from PIL import Image

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

max_sr = 96000

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

        sr = librosa.get_samplerate(wav_doc) 
        sr_to_max_sr_ratio = sr / max_sr

        #on veut transofrmer cet array en image

        plt.figure(figsize=(10, round(4*sr_to_max_sr_ratio)))
        #on veut enlever les marges blanches
        plt.axis('off')
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])

        librosa.display.specshow(feature, sr=sr, x_axis='time', y_axis='linear')
        
        image_path = f"{base_path}{wav_doc.rsplit('/', 1)[1][:-4]}.png"
        plt.savefig(image_path, bbox_inches=None, pad_inches=0)
        #plt.show()
        plt.close()

        # Adding a black padding on top of spectrograms corresponding to recordings with a sampling frequency < 96KHz
        # This way we ensure that sounds recorded with higher sampling frequencies won't look flatter than sounds 
        # recorded with lower sampling frequencies.
        img = Image.open(image_path)
        img_w, img_h = img.size
        background = Image.new('RGBA', (1000, 400), (0, 0, 0, 255))
        bg_w, bg_h = background.size
        offset = ((bg_w - img_w), round(400 - 400*sr_to_max_sr_ratio))
        background.paste(img, offset)
        background.save(image_path)

        #Il y a différentes fréquence d'échantillonage(sampling rate) (cela pour capter différents bruits). 
        #Le sampling rate vaut deux fois la valeur maximale captée.
        #Mais comme le sampling rate varie selon les enregistrements, on va normaliser les spectogrammes
        # sr = librosa.get_samplerate(wav_doc) 
        # print(sr)
        # librosa.display.specshow(feature, sr=sr, fmax = 48000) #sr est la frequence d'echantillonagemax multipliée par 2
        # plt.savefig(base_path + f"{wav_doc[:-4]}.png")
        # plt.show()
        


