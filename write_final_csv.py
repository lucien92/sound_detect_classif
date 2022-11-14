import csv


#paths

base_path = "/home/lucien/Documents/sound_detect_classif/src/data/Sons/csv_spectro"

path_to_data = "/home/lucien/Documents/sound_detect_classif/src/data/Sons/csv_wav/classic_data.csv"

#we write a csv replacinf the path to the .wav path by a path to the spectrogram image

with open(path_to_data, "r") as f:
    lines = f.readlines()
    with open(base_path + "/classic_data_spectro.csv", "w") as f2:
        for line in lines:
            line = line.split(",")
            line[0] = line[0][:-4] + ".png"
            f2.write(",".join(line))