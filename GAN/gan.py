#import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, Reshape, BatchNormalization
from keras.layers import LeakyReLU, Dropout, Flatten, UpSampling2D, Input, ZeroPadding2D
from keras import optimizers
from keras.datasets import mnist
#import numpy
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class GAN():
	def __init__(self):
		self.input_shape = (28, 28, 1)
		self.latent_dim = 100
		#input layer
		input = Input(shape = (100,))

		self.discriminator = self.make_discriminator()
		self.discriminator.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])

		self.generator = self.make_generator()
		#img layer
		img = self.generator(input)	

		self.combined = Model(input, self.discriminator(img))
		self.discriminator.trainable = False
		self.combined.compile(loss='binary_crossentropy', optimizer= 'adam')


	def generate_noise(self, amount):
		return np.random.normal(0, 1, (amount, self.latent_dim))

	def make_generator(self):
		model = Sequential()
		model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
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

	def make_discriminator(self):
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


	def train(self, epoch, save_interval):
		#manipulate trainning data
		(X_train, _), (_, _) = mnist.load_data()
		X_train = X_train / 127.5 - 1
		X_train = np.expand_dims(X_train, axis=3)

		size = len(X_train)
		true = np.ones((size, 1))
		false = np.zeros((size, 1))		
		for i in range(epoch):
			noise = self.generate_noise(len(X_train))
			false_image = self.generator.predict(noise)
			''' train discriminator '''
			self.discriminator.fit(X_train, true, batch_size = int(size / 10), epochs = 2)
			self.discriminator.fit(false_image, false, batch_size = int(size / 10), epochs = 2)
			'''train generator '''
			self.combined.fit(noise, true, batch_size = int(size / 100), epochs = 1)
			if (i % save_interval == 0):
				self.save_progress(i)

	def save_progress(self, epoch):
		r, c = 5, 5
		noise = self.generate_noise(r * c)
		gen_imgs = self.generator.predict(noise)
		print(gen_imgs[0].shape)
		# Rescale images 0 - 1

		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("images/mnist_%d.png" % epoch, format = 'png')
		print("saved")
		plt.close()
if __name__ == '__main__':
	g = GAN()
	g.train(50, 10)