import json
from math import exp
import random
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class Network:
	def __init__(self, train_set, train_labels, test_set, test_labels, config):
		self.train_set = train_set
		self.train_labels = train_labels
		self.test_set = test_set
		self.test_labels = test_labels

		self.alpha = config['learning_rate']

		self.I = config['input_nodes']
		self.J = config['hidden_nodes']
		self.K = config['output_nodes']

		self.epochs = config['epochs']
		self.batch_size = config['batch_size']

		self.weights_to_hidden = randomize_weights(self.J, self.I)
		self.weights_to_output = randomize_weights(self.K, self.J)

		self.save_weights()

		self.hidden_bias = np.ones((self.J, 1))
		self.output_bias = randomize_weights(self.K, 1)

		possibles = globals().copy()
		possibles.update(locals())
		self.hidden_activ = possibles.get(config['hidden_activation_function'])
		self.output_activ = possibles.get(config['output_activation_function'])
		self.hidden_activ_prime = possibles.get(config['hidden_activation_function'] + '_prime')
		self.output_activ_prime = possibles.get(config['output_activation_function'] + '_prime')

	def train (self):
		for i in range (self.epochs):
			#Load weights
			self.load_weights()

			for j in range (self.train_set.shape[0] / self.batch_size):

				#Get batch
				batch, labels = self.get_batch(self.train_set, self.train_labels)

				#Weight changes
				delta_w_to_hidden, delta_w_to_output = np.zeros((self.weights_to_hidden.shape)), np.zeros((self.weights_to_output.shape))
				delta_bias_hidden, delta_bias_output = np.zeros((self.hidden_bias.shape)), np.zeros((self.output_bias.shape))

				for i in range(batch.shape[0]):
					inst = transpose(batch[i])
					label = transpose(labels[i])

					#Feed forward
					h, h_a, z, z_a = self.feed_forward(inst)

					#Back prop
					loss_grad_to_output, delta_o, loss_grad_to_hidden, delta_h, cost = self.back_prop(inst, h, h_a, z, z_a, label)

					#Update deltas
					delta_w_to_hidden += loss_grad_to_hidden
					delta_w_to_output += loss_grad_to_output
					delta_bias_hidden += delta_h
					delta_bias_output += delta_o

				#Update weights
				self.weights_to_hidden -= self.alpha * ((1.0 / self.batch_size) * delta_w_to_hidden)
				self.weights_to_output -= self.alpha * ((1.0 / self.batch_size) * delta_w_to_output) 

			#Save weights
			self.save_weights()

	def feed_forward (self, x):
		#Input to hidden
		h = np.dot(self.weights_to_hidden, x) + self.hidden_bias
		h_a = self.hidden_activ(h)

		#Hidden to output
		z = np.dot(self.weights_to_output, h_a) + self.output_bias
		z_a = self.output_activ(z)

		return h, h_a, z, z_a

	def back_prop (self, x, h, h_a, z, z_a, y):
		#Cost for instance
		cost = 0.5 * (loss(y - z_a) ** 2)

		#Output back to hidden
		delta_o = (-1 * (y - z_a)) * self.output_activ_prime(z)
		loss_grad_to_output = np.dot( delta_o, h_a.T )

		#Hidden back to input
		delta_h = np.dot(self.weights_to_output.T, delta_o) * self.hidden_activ_prime(h)
		loss_grad_to_hidden = np.dot( delta_h, x.T )

		return loss_grad_to_output, delta_o, loss_grad_to_hidden, delta_h, cost

	def test(self):
		correct, total = 0, 0
		for sample in range(self.test_set.shape[0]):

			#Get batch and label
			batch = np.array([self.test_set[sample]]).T
			label = np.array([self.test_labels[sample]])

			#Forward pass
			h, h_a, z, z_a = self.feed_forward(batch)

			#Find accuracy
			predicted = np.argmax(z_a)
			actual = np.argmax(label)
			if (predicted == actual):
				correct += 1
			total += 1

		accuracy = str((float(correct) / float(total)) * 100) + '%'
		print accuracy

	def get_batch (self, inst, lab):
		batch = np.zeros((self.batch_size, inst.shape[1]))
		labels = np.zeros((self.batch_size, lab.shape[1]))
		for i in range (self.batch_size):
			r = random.randint(0, inst.shape[0] - 1)
			batch[i] = inst[r]
			labels[i] = lab[r]
		return batch, labels

	def save_weights (self):
		self.weights_to_hidden.dump('model/weights_to_hidden.dat')
		self.weights_to_output.dump('model/weights_to_output.dat')
	
	def load_weights (self):
		self.weights_to_hidden = np.load('model/weights_to_hidden.dat')
		self.weights_to_output = np.load('model/weights_to_output.dat')

def randomize_weights (a, b):
	weights = np.zeros((a, b))
	for i in range (a):
		for j in range (b):
			r = random.uniform(0, 0.6) - 0.3
			weights[i, j] = r 
	return weights

def sigmoid (i):
	return (1 / (1 + np.exp(-i)))

def ReLU (h):
	func = lambda x: x if x > 0 else np.array([0])
	return np.array([func(i) for i in h])

def sigmoid_prime (i):
	s = sigmoid(i)
	return np.multiply( s, (np.ones((s.shape)) - s) )

def ReLU_prime (h):
	func = lambda x: 1 if x > 0 else 0
	return transpose( np.array([func(i) for i in h]) )

def loss (a):
	return sum([i**2 for i in a.T[0]])

def transpose (x):
	return np.array(([x])).T

#Main
train_set = mnist.train.images
train_labels = mnist.train.labels 
test_set = mnist.test.images
test_labels = mnist.test.labels

with open ("config.json") as json_file:
	config = json.load(json_file)

nn = Network(train_set, train_labels, test_set, test_labels, config)
nn.train()
nn.test()