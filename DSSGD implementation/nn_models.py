#import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
#import numpy & sklearn
import numpy as np
from sklearn import metrics
#load data
from keras.datasets import mnist
from keras.utils import np_utils
#build model
def get_mlp_model():
	model = Sequential()
	model.add(Dense(units = 500, input_dim=784))
	model.add(Activation("relu"))
	model.add(Dense(units = 500, input_dim=784))
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

if __name__ == "__main__":
	test(get_mlp_model())