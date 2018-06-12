#import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
#import numpy & sklearn
import numpy as np

import nn_models

class Participant():
	def __init__(self, server, X_train, y_train):
		self.server = server
		self.X_train = X_train
		self.y_train = y_train
		self.models = nn_models.get_mlp_model()

	def train(self, num_of_epochs):
		batch_size = self.y_train.size / 100
		self.models.fit(self.X_train, self.y_train, epochs = num_of_epochs, batch_size = int(batch_size))

	def dssgd(self):
		'''
			train 1 epoch
			calculate the difference between before and after weights
			upload that difference to the server
		'''
		old_weights = self.models.get_weights()
		server_weights = self.server.download_weights(1)
		self.models.set_weights(server_weights)
		self.train(1)
		new_weights = self.models.get_weights()
		weight_changes = np.array(new_weights) - np.array(old_weights)
		self.server.receive_weights(weight_changes)


if __name__ == '__main__':
	X_train, y_train, X_test, y_test = nn_models.get_data()
	p = Participant(None ,X_train, y_train)
	p.train(5)