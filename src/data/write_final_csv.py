import csv
import PIL
from PIL import Image

#paths

base_path = "/home/acarlier/code/audio_recognition_yolo/src/data/Sons/csv_spectro"

path_to_data = "/home/acarlier/code/audio_recognition_yolo/src/data/Sons/csv_wav/classic_data.csv"

#we write a csv replacinf the path to the .wav path by a path to the spectrogram image

with open(path_to_data, "r") as f:
    lines = f.readlines()
    with open(base_path + "/classic_data_spectro.csv", "w") as f2:
        for line in lines:
            line = line.split(",")
            line[0] = line[0][:-4] + ".png"
            f2.write(",".join(line))
            
#on veut rajouter deux colonnes qui indiquent la taille de l'image renseignée dans le csv à line[0]
path_to_images = "/home/acarlier/code/audio_recognition_yolo/src/data/Spectrograms_nuls"

with open(base_path + "/classic_data_spectro.csv", "r") as f:
    lines = f.readlines()
    with open(base_path + "/classic_data_spectro.csv", "w") as f2:
        for line in lines:
            line = line.split(",")
            try:
                img = Image.open(path_to_images + "/" + line[0])
                width, height = img.size
                line.append(str(width))
                line.append(str(height) + '\n')
                line[5] = line[5].replace('\n', '')
                f2.write(",".join(line))
                
            except:
                print("error", line[0])
                pass
            # img = PIL.Image.open(path_to_images + "/" + line[0])
            # width, height = img.size
            # line.insert(1, str(width))
            # line.insert(2, str(height))
            # f2.write(",".join(line))
