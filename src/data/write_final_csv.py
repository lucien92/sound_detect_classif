import csv
import PIL
import math
import pandas as pd
import numpy as np
import librosa
from PIL import Image

#paths

base_path = "/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs"

path_to_data = "/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/classic_data.csv"

#we write a csv replacing the path to the .wav file with a path to the spectrogram image

splits_per_recording_df = pd.read_csv('/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/n_splits_per_recording.csv', delimiter=',')

with open(path_to_data, "r") as f:
    lines = f.readlines()
    with open(base_path + "/classic_data_spectro.csv", "w") as f2:
        for line in lines:
            line = line.split(",")

            xmin = float(line[1])
            xmax = float(line[3])
            split_min = math.trunc(xmin/5)
            split_max = math.trunc(xmax/5)
            new_line = line.copy()

            splits_in_current_recording = splits_per_recording_df[splits_per_recording_df['recording']==line[0]]['n_splits'].item()

            for i in range(split_min, split_max+1):
                if i < splits_in_current_recording:
                    new_line[0] = line[0][:-4] + "_split_" + str(i+1) + ".png"
                    new_line[1] = str(max(0, round(xmin - i*5, 2))) # xmin
                    new_line[2] = str(max(0, float(new_line[2]))) # When choosing a minimum frequency of 0 Hz on Audacity the annotations generated display a frequency of -1.0 instead of 0.0
                    new_line[3] = str(min(5, round(xmax - i*5, 2))) # xmax
                    # print(line)
                    # print(split_min)
                    # print(split_max)
                    # print(i)
                    # print(new_line)

                    f2.write(",".join(new_line))
            
#on veut rajouter deux colonnes qui indiquent la taille de l'image renseignée dans le csv à line[0]
path_to_images = "/home/david/Escriptori/Feines/sound_detect_classif/src/data/Spectrograms"
n_mels = 128

with open(base_path + "/classic_data_spectro.csv", "r") as f:
    lines = f.readlines()
    with open(base_path + "/classic_data_spectro.csv", "w") as f2:
        for line in lines:
            line = line.split(",")

            wav_doc = line[0].rsplit('_split_', 1)[0] + ".wav"
            sr = librosa.get_samplerate(wav_doc)
            freqs = librosa.core.mel_frequencies(fmin=0.0, fmax=sr / 4, n_mels=n_mels)

            line[0] = path_to_images + "/" + line[0].rsplit('/', 1)[1]
            try:
                img = Image.open(line[0])
                width, height = img.size
                line[1] = str(round(float(line[1]) / 5, 4))
                line[3] = str(round(float(line[3]) / 5, 4))
                # line[2] = str(round(float(line[2]) / 48000, 4))
                # line[4] = str(round(float(line[4]) / 48000, 4))


                # Transforming frequencies from Hz to the mel scalefind = 4400 
                line[2] = str(1 - round((np.argmin(abs(freqs - float(line[2]))) / n_mels) * 0.5, 4))
                line[4] = str(1 - round((np.argmin(abs(freqs - float(line[4]))) / n_mels) * 0.5, 4))
                line.append(str(width))
                line.append(str(height) + '\n')

                #on ne veut pas garder les chiffres allant de 1 à 10 dans le nom de l'image
                line[5] = line[5].replace('\n', '')
                line[5] = line[5].replace(' 0', '')
                line[5] = line[5].replace(' 1', '')
                line[5] = line[5].replace(' 2', '')
                line[5] = line[5].replace(' 3', '')
                line[5] = line[5].replace(' 4', '')
                line[5] = line[5].replace(' 5', '')
                line[5] = line[5].replace(' 6', '')
                line[5] = line[5].replace(' 7', '')
                line[5] = line[5].replace(' 8', '')
                line[5] = line[5].replace('Rana temporaria ', 'Rana temporaria')
                line[5] = line[5].replace('Hyla meridionalis ', 'Hyla meridionalis')
                line[5] = line[5].replace('Orage ', 'Orage')
                line[5] = line[5].replace('Deplacement', 'Déplacement')
                line[5] = line[5].replace('Orthoptere sp.', 'Orthoptera sp.')
                line[5] = line[5].replace('Tur mer', 'Turdus merula')
                line[5] = line[5].replace('Yersinella raymondi', 'Yersinella raymondii')
                line[5] = line[5].replace('Yersinella raymondiii', 'Yersinella raymondii')
                line[5] = line[5].replace('Luscinia megarhynchyos', 'Luscinia megarhynchos')
                line[5] = line[5].replace('Pseudochortippus parallelus', 'Pseudochorthippus parallelus')

                if not ('Non identifié' in line[5]):
                    f2.write(",".join(line))
                
            except:
                #print("error", line[0])
                pass
            # img = PIL.Image.open(path_to_images + "/" + line[0])
            # width, height = img.size
            # line.insert(1, str(width))
            # line.insert(2, str(height))
            # f2.write(",".join(line))
