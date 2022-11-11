#!/usr/bin/env python3
#https://colab.research.google.com/drive/1GSeBQdyZH_nHvl52XW0uhXV3Cslho24O#scrollTo=RyOYq9mv2ppC
#auf 5, 8, 8 aendern von 14,8,8 und tanh statt sigmoid im Vergleich zum Original

""" gtx 770 braucht anaconda umgebung mit altem tensorflow
#https://stackoverflow.com/questions/39023581/tensorflow-cuda-compute-capability-3-0-the-minimum-required-cuda-capability-is/59248949#59248949
#conda create -n tf-gpu
#conda activate tf-gpu
#conda install tensorflow-gpu=1.12
"""

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
import numpy

class Net():
	"""
		build_model(32, 4)
		print("build model")
		#Requirement
		# conda install graphviz
		# conda install pydot
		utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)
	"""
	def build_model(self, conv_size, conv_depth):
		board3d = layers.Input(shape=(5, 8, 8))

		# adding the convolutional layers
		x = board3d
		for _ in range(conv_depth):
			x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu', data_format='channels_first')(x)
		x = layers.Flatten()(x)
		x = layers.Dense(64, 'relu')(x)
		x = layers.Dense(1, 'tanh')(x)

		return models.Model(inputs=board3d, outputs=x)
	
	def build_model_residual(self, conv_size, conv_depth):
		board3d = layers.Input(shape=(5, 8, 8))

		# adding the convolutional layers
		x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_first')(board3d)
		for _ in range(conv_depth):
			previous = x
			x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_first')(x)
			x = layers.BatchNormalization()(x)
			x = layers.Activation('relu')(x)
			x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_first')(x)
			x = layers.BatchNormalization()(x)
			x = layers.Add()([x, previous])
			x = layers.Activation('relu')(x)
		x = layers.Flatten()(x)
		x = layers.Dense(1, 'tanh')(x)

		return models.Model(inputs=board3d, outputs=x)
	  


# datatset mit Dateipfad /processed/.. reinladen und container mit arr_0 und arr_1 bennenen
import tensorflow.keras.callbacks as callbacks

class Dataset():
	def get_dataset(self):
		container = numpy.load('./processed/dataset_10M.npz')
		b, v = container['arr_0'], container['arr_1']
		# v = numpy.asarray(v / abs(v).max() / 2 + 0.5, dtype=numpy.float32) # normalization (0 - 1)
		return b, v #ram laeuft ueber, also erstmal 2M/25M

if __name__== "__main__":
	net = Net()
	model = net.build_model_residual(32, 4)
	utils.plot_model(model, to_file='model_plot_5_8_8.png', show_shapes=True, show_layer_names=False)
	chessDataset=Dataset()
	x_train, y_train = chessDataset.get_dataset()
	print(x_train.shape)
	print(y_train.shape)
	

	model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')
	model.summary()
	model.fit(x_train, y_train,
			  batch_size=2048,
			  epochs=100,
			  verbose=1,
			  validation_split=0.1,
			  callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
						 callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)])

	model.save('model_10M_5_8_8_residual.h5')