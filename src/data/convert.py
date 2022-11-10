import os
import csv

base_path = '/home/acarlier/code/audio_recognition_yolo/src/data/Sons/'

for file in os.listdir(base_path):
    if file.endswith('.txt'):
        print(file)
        with open(base_path + file, 'r') as f:
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
            
            
        try:
            os.mkdir(base_path + 'csv/')
        except:
            pass
        
        
        # with open('/home/acarlier/code/audio_recognition_yolo/src/data/Sons/csv/fake_classic_data.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     for i in range(len(new_lines)):
        #         writer.writerow([new_lines[i]])
                
with open('/home/acarlier/code/audio_recognition_yolo/src/data/Sons/csv/fake_classic_data.csv', 'r') as f:
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
                
with open('/home/acarlier/code/audio_recognition_yolo/src/data/Sons/csv/classic_data.csv', 'w') as f:
    writer = csv.writer(f)# quoting = csv.QUOTE_NONE, escapechar=' ')
    #on veut écrire les élements de new_lines dans le fichier csv
    for i in range(len(new_lines)):
        #on veut écrire line sans guillemets sur le csv
        writer.writerow([new_lines[i]])


            
            
        
        