#import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
#import numpy & sklearn
import numpy as np

from participant import Participant

import nn_models

class Malicious_Participant(Participant):
	def __init__(self, server, X_train, y_train, upload_percentage):
		super.__init__(server, X_train, y_train, upload_percentage)
		self.generator = nn_models.make_generator()
		self.target = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
		