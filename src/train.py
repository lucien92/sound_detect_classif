import math
import random
import numpy as np

from .config import config
from .dataset.clips_dataset import clips_dataset
from .data.generator import Generator
# from .networks.cifar10 import create_model
from .networks.resnet import create_model

## DATA
# Load dataset
dataset = clips_dataset()
labels = np.unique([dataset[i]['name'] for i in range(len(dataset))])

random.Random(1337).shuffle(dataset)
validation_samples = math.floor(len(dataset) * 0.2)
train_dataset = dataset[:-validation_samples]
validation_dataset = dataset[-validation_samples:]

# Create data generator (features extraction + normalization + augmentation)
train = Generator(train_dataset, 5, labels)
validation = Generator(validation_dataset, 5, labels)

## Model
input_shape = train.input_shape
n_classes = train.n_classes
model = create_model(input_shape, n_classes)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train, epochs=config['epochs'], validation_data=validation, steps_per_epoch=10, validation_steps=2)
# model.fit(train, epochs=config['epochs'], validation_data=validation, steps_per_epoch=10, validation_steps=2)