#import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
#import numpy & sklearn
import numpy as np

import nn_models

class Participant():
	def __init__(self, server, X_train, y_train, upload_percentage):
		self.server = server
		self.X_train = X_train
		self.y_train = y_train
		self.models = nn_models.get_mlp_model()
		self.models.set_weights(self.server.download_weights(1))
		self.upload_percentage = upload_percentage
		self.train(5)

	def train(self, num_of_epochs):
		batch_size = self.y_train.size / 100
		self.models.fit(self.X_train, self.y_train, epochs = num_of_epochs, batch_size = int(batch_size))

	def dssgd(self):
		'''
			train 1 epoch
			calculate the difference between before and after weights
			upload that difference to the server
		'''
		server_weights = self.server.download_weights(1)
		self.models.set_weights(nn_models.replace_respective_values(server_weights, self.models.get_weights()))
		old_weights = self.models.get_weights()
		self.train(1)
		new_weights = self.models.get_weights()
		weight_changes = np.array(new_weights) - np.array(old_weights)
		#only upload the desired percentage of weight_changes
		self.server.receive_weights(nn_models.return_matrix_with_greatest(weight_changes, self.upload_percentage))


if __name__ == '__main__':
	X_train, y_train, X_test, y_test = nn_models.get_data()
	p = Participant(None ,X_train, y_train)
	p.train(5)