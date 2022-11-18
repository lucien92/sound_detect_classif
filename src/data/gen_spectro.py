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

file_list = []

with open("/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/classic_data.csv", "r") as f:
    lines = f.readlines()
    for line in lines:
        file_list.append(line.split(",")[0])

# Preserving only one copy of each file name
file_list = list(set(file_list))
file_list.sort()

for wav_doc in file_list:
    
    print(wav_doc)
    waveform, _ = librosa.load(wav_doc) 

    sr = librosa.get_samplerate(wav_doc) 
    sr_to_max_sr_ratio = sr / max_sr
    
    # Get number of samples for 5 seconds
    buffer = round(5 * sr/2)

    total_samples = len(waveform)
    saved_samples = 0
    counter = 1

    while (saved_samples + buffer) < total_samples:

        #check if the buffer is not exceeding total samples 
        if buffer > (total_samples - saved_samples):
            buffer = total_samples - saved_samples

        block = waveform[saved_samples : (saved_samples + buffer)]
        suffix = "_split_" + str(counter)

        feature = spectrogram(block, 512, 256) #nfft et hop_length, nfft est la taille de la fenetre, hop_length est le pas entre deux fenetres



        #on veut transofrmer cet array en image

        fig_width = 10
        fig_height = 20*sr_to_max_sr_ratio
        
        plt.figure(figsize=(fig_width, fig_height))
        #on veut enlever les marges blanches
        plt.axis('off')
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])

        librosa.display.specshow(feature, sr=sr, x_axis='time', y_axis='linear')
        
        image_path = f"{base_path}{wav_doc.rsplit('/', 1)[1][:-4]}{suffix}.png"
        plt.savefig(image_path, bbox_inches=None, pad_inches=0)
        #plt.show()
        plt.close()

        # Adding a black padding on top of spectrograms corresponding to recordings with a sampling frequency < 96KHz
        # This way we ensure that sounds recorded with higher sampling frequencies won't look flatter than sounds 
        # recorded with lower sampling frequencies.
        img = Image.open(image_path)
        img_w, img_h = img.size

        # Image size = 1000 x 400 px
        background = Image.new('RGBA', (round(fig_width * 100), 2000), (0, 0, 0, 255))
        bg_w, bg_h = background.size

        # Ensuring that the resulting image will have a size of 1000 x 400 px
        offset = ((bg_w - img_w), round(2000 - 2000*sr_to_max_sr_ratio))
        background.paste(img, offset)
        background.save(image_path)

        counter += 1
        saved_samples += buffer


    

        


