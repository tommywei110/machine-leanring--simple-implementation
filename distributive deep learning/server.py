#import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
#import numpy & sklearn
import numpy as np
#get model
import nn_models as nn_models
#get participant
from participant import Participant

class Server():
	num_of_participants = 0
	participants = []
	history_acc = []
	def __init__(self, num_of_participants, decade_rate, upload_percentage):
		self.num_of_participants = num_of_participants
		self.upload_percentage = upload_percentage
		self.decade_rate = decade_rate
		self.model = nn_models.get_mlp_model()
		w = np.array(self.model.get_weights())
		self.stats_matrix = w - w + 1
		self.X_train, self.y_train, self.X_test, self.y_test = nn_models.get_data()
		if num_of_participants != 10:
			self.split_data()

		else:
			#each participant has a unique labeled training set
			self.data_catagorize()
		self.assign_data_to_participants()
		score = self.model.evaluate(self.X_test, self.y_test)
		self.history_acc.append(score[1])


	def split_data(self):
		self.X_train = np.split(self.X_train, self.num_of_participants)
		self.y_train = np.split(self.y_train, self.num_of_participants)

	def data_catagorize(self):
		new_X = [[] for _ in range(10)]
		new_Y = [[] for _ in range(10)]
		for i in range(6000):
			for j in range(10):
				if self.y_train[i][j] == 1:
					new_X[j].append(self.X_train[i].copy())
					new_Y[j].append(self.y_train[i].copy())
		for i in range(10):
			new_X[i] = np.array(new_X[i])
			new_Y[i] = np.array(new_Y[i])
		self.X_train = new_X
		self.y_train = new_Y

	def assign_data_to_participants(self):

		for i in range(self.num_of_participants):
			self.participants.append(Participant(self, self.X_train[i], self.y_train[i], self.upload_percentage))

	def download_weights(self, theta):
		#download theta percent of weights with the highest stat	
		only_the_biggest_stats = nn_models.return_matrix_with_greatest(self.stats_matrix, theta)
		w = self.model.get_weights()
		return nn_models.respective_values(only_the_biggest_stats, w)

	def receive_weights(self, weight_changes):
		current_weights = self.model.get_weights()
		new_weights = np.array(current_weights) + weight_changes
		update_stats = nn_models.respective_values(weight_changes, self.stats_matrix)
		update_stats = update_stats * self.decade_rate
		self.stats_matrix = nn_models.replace_respective_values(update_stats, self.stats_matrix)
		self.model.set_weights(new_weights.tolist())

	

	def server_epoch(self):
		for p in self.participants:
			p.dssgd()
			score = self.model.evaluate(self.X_test, self.y_test)
		self.history_acc.append(score[1])
	

if __name__ == '__main__':
	s = Server(5,0.9,0.5)
	for _ in range(3):
		s.server_epoch()
	r = s.model.predict(s.X_test)
	print(r[0])

