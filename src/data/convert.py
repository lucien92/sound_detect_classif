import os
import csv
import glob
import itertools

#base_path = '/media/david/One Touch/PSI-BIOM/Travail/Enregistrements opportunistes/'
base_path = '/media/david/One Touch/Sons/'
annotations = []

try:
    os.mkdir('/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/')
except:
    pass

for file in glob.iglob(base_path + '**/*.txt', recursive=True):
#for file in os.listdir(base_path):
    matches = ["log.txt", "Summary", "Readme"]

    if not any(x in file for x in matches):
        print(file)
        with open(file, 'r') as f:
            lines = f.readlines()
            i = 1
            new_lines = []
            while i < len(lines):
                new_lines.append(lines[i-1] + ',' + lines[i])
                i += 2
                # print(new_lines)
            index_to_delete = []
            for i in range(len(new_lines)):
                
                new_lines[i] = new_lines[i].replace('\t', ',')
                new_lines[i] = new_lines[i].replace('\n', '')
                new_lines[i] = new_lines[i].replace('\\', '')
                new_lines[i] = new_lines[i].replace(',,', ',')
                new_lines[i] = file[:-4] + '.wav' + ',' + new_lines[i]
                
                
                if ' 1 ' in new_lines[i]:
                    
                    new_lines[i] = new_lines[i].replace(' 1 1', '')
                    new_lines[i] = new_lines[i].replace(' 1 2', '')
                    new_lines[i] = new_lines[i].replace(' 1 3', '')
                    new_lines[i] = new_lines[i].replace(' 1 4', '')
                    
                
                if ' 0 ' in new_lines[i]:
                    index_to_delete.append(i)

                    
            for i in sorted(index_to_delete, reverse=True):
                del new_lines[i]
               
            # with open('/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/temp_classic_data.csv', 'a') as f:
            #     writer = csv.writer(f)
            #     for i in range(len(new_lines)):
            #         writer.writerow([new_lines[i]])
            for i in range(len(new_lines)):
                annotations.append(new_lines[i])


with open('/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/temp_classic_data.csv', 'w') as f:
    writer = csv.writer(f)

    for line in annotations:
        print(line.split(','))
        writer.writerow(line.replace(" ,", ",").split(','))              
                
with open('/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/temp_classic_data.csv', 'r') as f:
    lines = f.readlines()
    new_lines = []
    for line in lines:
        line = line.split(',')
        x =  round(float(line[1]),2)
        y = round(float(line[2]),2)
        z = round(float(line[4]),2)
        line[5] = line[5][:-2]
        w = round(float(line[5]),2)
        #fname, xmin, ymin, xmax, ymax, class, width, height
        new_lines.append(line[0] + ',' + str(x) + ',' + str(z) + ',' + str(y) + ',' + str(w) + ',' + line[3])
        #print("hello", len(new_lines))
                
with open('/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/classic_data.csv', 'w') as f:
    writer = csv.writer(f)# quoting = csv.QUOTE_NONE, escapechar=' ')
    #on veut écrire les élements de new_lines dans le fichier csv
    for i in range(len(new_lines)):
        #on veut écrire line sans guillemets sur le csv
        writer.writerow(new_lines[i].split(','))

try:
    os.remove('/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/temp_classic_data.csv')
except OSError:
    pass
            
            
        
        