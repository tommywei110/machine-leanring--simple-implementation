#import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
#import numpy & sklearn
import numpy as np
#get model
import nn_models
#get participant
from participant import Participant
class Server():
	num_of_participants = 0
	participants = []
	history_acc = []
	def __init__(self, num_of_participants):
		self.num_of_participants = num_of_participants
		self.model = nn_models.get_mlp_model()
		self.stats_matrix = self.model.get_weights()
		self.X_train, self.y_train, self.X_test, self.y_test = nn_models.get_data()
		self.split_data()
		self.assign_data_to_participants()
		self.initial_train()
		score = self.model.evaluate(self.X_test, self.y_test)
		self.history_acc.append(score[1])

	def split_data(self):
		self.X_train = np.split(self.X_train, self.num_of_participants)
		self.y_train = np.split(self.y_train, self.num_of_participants)

	def assign_data_to_participants(self):
		for i in range(self.num_of_participants):
			self.participants.append(Participant(self, self.X_train[i], self.y_train[i]))

	def initial_train(self):
		for p in self.participants:
			p.train(5)

	def download_weights(self, theta):
		#download theta percent of weights with the highest stat	
		return self.model.get_weights()

	def receive_weights(self, weight_changes):
		current_weights = self.model.get_weights()
		new_weights = np.array(current_weights) + weight_changes
		self.model.set_weights(new_weights.tolist())

	def server_epoch(self):
		for p in self.participants:
			p.dssgd()
			score = self.model.evaluate(self.X_test, self.y_test)
		self.history_acc.append(score[1])
	

if __name__ == '__main__':
	s = Server(3)
	for i in range(5):
		s.server_epoch()
	import matplotlib.pyplot as plt
	x = range(len(s.history_acc))
	plt.plot(x,s.history_acc)
	plt.show()
