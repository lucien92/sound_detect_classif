from keras_yolov2.utils import draw_boxes, BoundBox
import cv2
import csv
import argparse
import json
import os
import tensorflow as tf
#on parse

argparser = argparse.ArgumentParser(
  description='Predict tflite model in real time or with movies / images')

argparser.add_argument(
  '-c',
  '--conf',
  default='/home/david/Escriptori/Feines/sound_detect_classif/src/config/benchmark_config/audio_classic.json',
  type=str,
  help='path to configuration file')

def _main_(args):
    
    #config

    with open(args.conf) as config_buffer:
        config = json.load(config_buffer)

    path_to_images = "/home/david/Escriptori/Feines/sound_detect_classif/src/data/Spectrograms/Anura_blantec31FR_01VI2022_i158_split_1.png"

    #coordonnées bounding boxes

    with open("/home/david/Escriptori/Feines/sound_detect_classif/src/data/CSVs/classic_data_spectro.csv", "r") as f:
        reader = csv.reader(f)
        for lines in reader:
            if lines[0] == path_to_images:
                print(lines)
                x1 = float(lines[1])
                y1 = float(lines[2])
                x2 = float(lines[3])
                y2 = float(lines[4])
                
                frame = cv2.imread(path_to_images)
                
                # Draw bounding boxes
                frame = draw_boxes(frame, [BoundBox(x1, y1, x2, y2, 1, config['model']['labels'])], config['model']['labels'])
                cv2.imshow('frame', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
               

if __name__ == '__main__':
  _args = argparser.parse_args()
  gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
  with tf.device('/GPU:' + gpu_id):
    _main_(_args)