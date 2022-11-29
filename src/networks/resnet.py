import tensorflow as tf

def create_model(input_shape, n_classes):
	return tf.keras.applications.ResNet50(
		include_top=True,
		weights=None,
		input_tensor=None,
		input_shape=input_shape,
		pooling=None,
		classes=n_classes
	)