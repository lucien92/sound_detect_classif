from keras.layers import Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential

def create_model(input_shape, nb_classes):
	channel_axis = 3

	model = Sequential()
	model.add(Conv2D(filters=32, input_shape=input_shape, padding='same', kernel_size=(3, 3)))
	model.add(BatchNormalization(axis=channel_axis))
	model.add(Activation('elu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	for layer in range(2):
		model.add(Conv2D(filters=32, padding='same', kernel_size=(3, 3)))
		model.add(BatchNormalization(axis=channel_axis))
		model.add(Activation('elu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(rate=0.25))

	model.add(Flatten())
	model.add(Dense(units=128))
	model.add(Activation('elu'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=nb_classes))
	model.add(Activation('softmax'))
	return model