from copyreg import pickle
import matplotlib.pyplot as plt
import argparse
import json
import os
import pickle


argparser = argparse.ArgumentParser(
    description='Plot training loss and validation loss hisotiry.')

argparser.add_argument(
    '-c',
    '--conf',
    default='/home/acarlier/project_ornithoScope_lucien/src/config/benchmark_config/test_imagenet_finetune.json',
    help='Path to config file.')


def _main_(args):
    config_path = args.conf

    # Load config file as a dict
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    
    # Get the pickle history file based on config file
    root, ext = os.path.splitext(config['train']['saved_weights_name'])
    saved_pickle_path = config['data']['saved_pickles_path']
    pickle_path = f'{saved_pickle_path}/history/history_{root}_bestLoss{ext}.p'

    # Load history pickle
    with open(pickle_path, 'rb') as pickle_buffer:
        history = pickle.load(pickle_buffer)
        
    # Extract losses

    loss = history['loss']
    val_loss = history['val_loss']

    for i in range(len(loss)):
        if loss[i] > 100:
            loss[i] = 100
        if val_loss[i] > 100:
            val_loss[i] = 100

    steps = [i for i in range(len(loss))]

    # Plot curves
    plt.plot(steps, loss, label='Training loss')
    plt.plot(steps, val_loss, label='Validation loss')
    
    # Modify figure params
    _, xmax, _, _ = plt.axis()
    xmin = int(0.1 * xmax)
    plt.axis((xmin, xmax, min(loss + val_loss), val_loss[xmin]))
    plt.legend()

    # Save figure
    plt.savefig(pickle_path + '.jpg')


if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)
