import csv

with open("/home/acarlier/code/audio_recognition_yolo/src/data/Sons/csv_spectro/classic_data_spectro.csv", 'r') as f:
    line = f.readlines()
    train = line[:int(len(line)*0.8)]
    val = line[int(len(line)*0.8):int(len(line)*0.9)]
    test = line[int(len(line)*0.9):]

list_espece = []
with open("/home/acarlier/code/audio_recognition_yolo/src/data/Sons/csv_spectro/train.csv", 'w') as f:
    writer = csv.writer(f)
    for i in range(len(train)):
        espece = train[i].split(',')[5]
        if espece not in list_espece:
            list_espece.append(espece)
        train[i] = train[i].replace('\n', '')
        writer.writerow([train[i]])
print(list_espece)
        
with open("/home/acarlier/code/audio_recognition_yolo/src/data/Sons/csv_spectro/val.csv", 'w') as f:
    writer = csv.writer(f)
    for i in range(len(val)):
        val[i] = val[i].replace('\n', '')
        writer.writerow([val[i]])
        
with open("/home/acarlier/code/audio_recognition_yolo/src/data/Sons/csv_spectro/test.csv", 'w') as f:
    writer = csv.writer(f)
    for i in range(len(test)):
        test[i] = test[i].replace('\n', '')
        writer.writerow([test[i]])

