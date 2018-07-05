#import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, UpSampling2D, Conv2D, BatchNormalization, Input
from keras import optimizers
#import numpy & sklearn
import numpy as np
from sklearn import metrics
#load data
from keras.datasets import mnist
from keras.utils import np_utils
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#build model
def get_mlp_model():
	model = Sequential()
	model.add(Dense(units = 128, input_dim=784))
	model.add(Activation("relu"))
	model.add(Dense(units = 64))
	model.add(Activation("relu"))
	model.add(Dense(units = 10))
	model.add(Activation("softmax"))
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return model

def get_data():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.reshape(-1, 784)
	X_train = X_train/255
	X_test = X_test.reshape(-1, 784)
	X_test = X_test/255
	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)
	return X_train, y_train, X_test, y_test

def test(model):
	X_train, y_train, X_test, y_test = get_data()
	model.fit(X_train, y_train, epochs = 5, batch_size = 1000)
	#visualize result of the model
	y_predict = model.predict(X_test)
	y_true = np.argmax(y_test, axis = 1)
	y_pred = np.argmax(y_predict, axis = 1)
	model_mlp = model
	print('MLP result (one epoch):')
	print(metrics.classification_report(y_true, y_pred))

def return_matrix_with_greatest(array, beta):
	flattened_array = powerful_flatten(array)
	n = int(len(flattened_array) * beta)
	argArray = flattened_array.argsort()[-n:]
	flat_copy = flattened_array.copy()
	for i in argArray:
		flattened_array[i] *= 2
	flattened_array = flattened_array - flat_copy
	return flatten_to_matrix(array, flattened_array)

def flatten_to_matrix(model, array):
	result_array = []
	total = 0
	for m in model:
		size = m.size
		flat = array[total: total+size]
		temp = np.reshape(flat, m.shape)
		result_array.append(temp)
		total += size
	return np.array(result_array)
	
def respective_values(positions, values):
		'''
			return the values at positions where positions array has value
		'''
		flat_pos = powerful_flatten(positions)
		flat_val = powerful_flatten(values)
		flat_result = np.where(flat_pos != 0, flat_val, flat_pos)
		return flatten_to_matrix(positions, flat_result)

def replace_respective_values(source, target):
	flat_source = powerful_flatten(source)
	flat_target = powerful_flatten(target)
	flat_result = np.where(flat_source != 0, flat_source, flat_target)
	return flatten_to_matrix(target, flat_result)

def powerful_flatten(array):
	master_array = np.array([])
	for a in array:
		master_array = np.append(master_array, a.flatten())
	return master_array

def make_generator():
		model = Sequential()
		model.add(Dense(128 * 7 * 7, activation="relu", input_dim= 100))
		model.add(Reshape((7, 7, 128)))
		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(UpSampling2D())
		model.add(Conv2D(64, kernel_size=3, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("relu"))
		model.add(Conv2D(1, kernel_size=3, padding="same"))
		model.add(Activation("tanh"))
		input = Input(shape = (100, ))
		img = model(input)
		return Model(input, img)

def make_discriminator():
		model = Sequential()
		model.add(Conv2D(32, kernel_size=3, strides=2, input_shape= (28, 28, 1), padding="same"))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
		model.add(ZeroPadding2D(padding=((0,1),(0,1))))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))

		img = Input(shape= (28, 28, 1))
		validity = model(img)

		return Model(img, validity)

if __name__ == "__main__":
	m = make_generator()
