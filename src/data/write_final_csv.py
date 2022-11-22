import csv
import PIL
import math
import pandas as pd
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

with open(base_path + "/classic_data_spectro.csv", "r") as f:
    lines = f.readlines()
    with open(base_path + "/classic_data_spectro.csv", "w") as f2:
        for line in lines:
            line = line.split(",")
            line[0] = path_to_images + "/" + line[0].rsplit('/', 1)[1]
            try:
                img = Image.open(line[0])
                width, height = img.size
                line[1] = str(round(float(line[1]) * width / 5))
                line[3] = str(round(float(line[3]) * width / 5))
                line[2] = str(round(float(line[2]) * height / 48000))
                line[4] = str(round(float(line[4]) * height / 48000))
                line.append(str(width))
                line.append(str(height) + '\n')
                line[5] = line[5].replace('\n', '')
                f2.write(",".join(line))
                
            except:
                #print("error", line[0])
                pass
            # img = PIL.Image.open(path_to_images + "/" + line[0])
            # width, height = img.size
            # line.insert(1, str(width))
            # line.insert(2, str(height))
            # f2.write(",".join(line))
