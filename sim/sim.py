# this file represents a locally-run recreation of the study done in the paper Communication-Efficient Learning of Deep Networks from Decentralized Data, which will hencforth be refered to a s teh FedAvg paper

import torch
import numpy as np
import random
import json
import os
from tqdm import tqdm
from types import SimpleNamespace
from os.path import exists as file_exists

import sys

arg_list = sys.argv[1:]
arg_dict = {}
for index, arg in enumerate(arg_list):
	if (arg[0] == '-'):
		arg_dict[arg[1:]] = arg_list[index+1]

args = SimpleNamespace(**arg_dict)

sys.path.append('..')
from data.fashion_data import train_images, train_labels, test_images, test_labels
from data.MNIST_data import img_x_train, img_x_test, flat_x_train, flat_x_test, y_train, y_test

print('importing data')

# x_train = flat_x_train.reshape([-1, 784])
# x_test = flat_x_test.reshape([-1, 784])
# y_train = y_train.reshape([-1])
# y_test = y_test.reshape([-1])

x_train = img_x_train.reshape([-1, 1, 28, 28])
x_test = img_x_test.reshape([-1, 1, 28, 28])
y_train = y_train.reshape([-1])
y_test = y_test.reshape([-1])

# x_train = train_images
# x_test = test_images
# y_train = train_labels
# y_test = test_labels

from networks import FashionNet, ConvNet, ThreeNN
model_class = ConvNet

criterion = torch.nn.CrossEntropyLoss()
num_samples = 30_000
results_folder = './Sep_29_meeting/mnist_cnn_data'

# this size of batches during testing
TEST_BATCH_SIZE = 32
# the size of batches during client updates
B = 32
# number of clients
K = int(args.k)
# number of iid data samples assigned to each client
n_k = num_samples // K
# the number of passes over their data set that each client makes
E = 1
# fraction of clients queried per communication round
C = 1
# the learning rate
eta = 0.0001
# the satisfaction ratio
sat_ratio = float(args.sr)

torch.cuda.empty_cache()
device = 'cuda:0'
# device = 'cpu'

# calculates the accuracy score of a prediction y_hat and the ground truth y
I = torch.eye(10,10)
def get_accuracy(y_hat, y):
	y_vec = torch.tensor([I[int(i)].tolist() for i in y]).to(device)
	dot = torch.dot(y_hat.flatten(), y_vec.flatten())
	return dot/torch.sum(y_hat)

# gets the accuracy and loss of net on testing data
def val_evaluation(net):

	NUM_TEST_BATCHES = x_test.shape[0]//TEST_BATCH_SIZE

	loss = 0
	acc = 0

	for i in range(NUM_TEST_BATCHES):
		x_batch = x_test[TEST_BATCH_SIZE*i : TEST_BATCH_SIZE*(i+1)].to(device)
		y_batch = y_test[TEST_BATCH_SIZE*i : TEST_BATCH_SIZE*(i+1)].to(device)

		y_hat = net.forward(x_batch)

		loss += criterion(y_hat, y_batch).item()
		acc += get_accuracy(y_hat, y_batch).item()
	
	# normalizing the loss and accuracy
	loss = loss/NUM_TEST_BATCHES
	acc = acc/NUM_TEST_BATCHES

	return loss, acc

# this function performs the ClientUpdate regime as described in the FedAvg paper
def client_update(client):
	net = client['net']
	optimizer = client['optimizer']	
	indices = client['data_indices']

	# the data allocated to this worker
	x = x_train[indices]
	y = y_train[indices]

	NUM_BATCHES = x.shape[0] // B

	for i in range(E):
		for j in range(NUM_BATCHES):

			x_batch = x[B*j: B*(j+1)]
			y_batch = y[B*j: B*(j+1)]

			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)

			y_hat = net(x_batch)

			loss = criterion(y_hat, y_batch)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

def parameter_divergence(param_list, weights):

def weighted_average_parameters(param_list, weights):

	num_clients = len(param_list)
	avg_params = []

	for i, params in enumerate(param_list):

		if (i == 0):
			for p in params:
				avg_params.append(p.clone()*weights[i])

		else:
			for j, p in enumerate(params):
				avg_params[j].data += p.data*weights[i]

	return avg_params

# sets the parameters of a neural net
def set_parameters(net, params):
	current_params = list(net.parameters())
	for i, p in enumerate(params): 
		current_params[i].data = p.data.clone()

def training_run(num_global_updates):
	# print('instantiating networks')
	central_model = model_class().to(device)
	accs = []

	# each worker is characterized by a neural network, optimizer and allocation of data
	clients = []
	for i in range(K):
		net = model_class().to(device)
		set_parameters(net, central_model.parameters())

		client_dict = {
			'net': net,
			'optimizer': torch.optim.Adam([{'params': net.parameters()}], lr=eta),
			'data_indices': torch.randperm(x_train.shape[0])[0: n_k]
			# 'data_indices': torch.randperm(x_train.shape[0])[0: random.randint(round(0.5*n_k), round(1.5*n_k))]
		}

		clients.append(client_dict)

	for i in range(num_global_updates):
		# print('--------------------------------------------------------------------------------')
		# print('global training cycle', i+1)

		# randomly selecting clients
		client_indices = random.sample(list(range(K)), k=round(C*K))
		S = [clients[i] for i in client_indices]
		live_indices = []

		# print('performing client updates with allocations of: '+str([len(k['data_indices']) for k in S]))
		for i, k in enumerate(S):
			if (random.uniform(0, 1) < sat_ratio):
				# sends parameters to each selected client
				set_parameters(k['net'], central_model.parameters())

				# shuffles data
				random.shuffle(k['data_indices'])

				# client updates these parameters on their local dataset
				client_update(k)

				live_indices.append(i)

		live_workers = [S[i] for i in live_indices]
		param_list = [k['net'].parameters() for k in live_workers]
		weights = [len(k['data_indices']) for k in live_workers]
		weights = [w/sum(weights) for w in weights]
		avg_params = weighted_average_parameters(param_list, weights)

		# setting central model to the average of the parameters of the clients' models
		set_parameters(central_model, avg_params)

		# print('assessing aggregated parameters')
		loss, acc = val_evaluation(central_model)
		accs.append(acc)
		# print('loss | acc: ', round(loss, 2), '|', round(100*acc, 1))

	return accs

if (__name__ == '__main__'):

	num_trials = int(args.t)
	# a list of accs across training runs
	tr_accs = []

	for i in range(num_trials):
		print('training run')
		accs = training_run(5)

		tr_accs.append(accs)

		if not hasattr(args, 'f'):
			print([a[-1] for a in accs])

	if hasattr(args, 'f'):
		filename = args.f
		filepath = os.path.join(results_folder, filename)
		
		# print(filepath)
		# if file_exists(filepath):
		# 	print('appending data')
		# 	with open(filepath, 'r') as f:
		# 		old_accs = json.loads(f.read())
		# 		tr_accs = old_accs + tr_accs
		# else:
		# 	print('saving data')

		with open(filepath, 'w') as f:
			f.write(json.dumps(tr_accs))