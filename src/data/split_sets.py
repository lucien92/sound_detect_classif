import csv
import pandas as pd
import json
import os

all_labels_df = pd.read_csv('/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/All_sound_categories.csv')

with open("/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/classic_data_spectro.csv", 'r') as f:
    line = f.readlines()
    train = line[:int(len(line)*0.8)]
    val = line[int(len(line)*0.8):int(len(line)*0.9)]
    test = line[int(len(line)*0.9):]

species_list = []
taxonomic_groups_list = []

with open("/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/train.csv", 'w') as f:
    writer = csv.writer(f)
    for i in range(len(train)):
        species = train[i].split(',')[5]
        # print("_" + species + "_")
        taxonomic_group = all_labels_df.loc[all_labels_df['Label'] == species]['Taxonomic_group'].values[0]
        # print(taxonomic_group)
        if species not in species_list:
            species_list.append(species)
        if taxonomic_group not in taxonomic_groups_list:
            taxonomic_groups_list.append(taxonomic_group)
        train[i] = train[i].replace('\n', '')
        # train[i] = train[i].replace(species, taxonomic_group)
        writer.writerow(train[i].split(','))
print(species_list)
print(taxonomic_groups_list)
        
with open("/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/val.csv", 'w') as f:
    writer = csv.writer(f)
    for i in range(len(val)):
        species = val[i].split(',')[5]
        # print("_" + species + "_")
        taxonomic_group = all_labels_df.loc[all_labels_df['Label'] == species]['Taxonomic_group'].values[0]
        # print(taxonomic_group)
        val[i] = val[i].replace('\n', '')
        # val[i] = val[i].replace(species, taxonomic_group)
        writer.writerow(val[i].split(','))
        
with open("/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/test.csv", 'w') as f:
    writer = csv.writer(f)
    for i in range(len(test)):
        species = test[i].split(',')[5]
        # print("_" + species + "_")
        taxonomic_group = all_labels_df.loc[all_labels_df['Label'] == species]['Taxonomic_group'].values[0]
        # print(taxonomic_group)
        test[i] = test[i].replace('\n', '')
        # test[i] = test[i].replace(species, taxonomic_group)
        writer.writerow(test[i].split(','))


filename = '/home/david/Escriptori/Feines/sound_detect_classif/src/config/benchmark_config/audio_classic.json'
with open(filename, 'r') as f:
    data = json.load(f)
    # data['model']['labels'] = taxonomic_groups_list
    data['model']['labels'] = species_list

os.remove(filename)
with open(filename, 'w') as f:
    json.dump(data, f, indent=4)