# this file represents a locally-run recreation of the study done in the paper Communication-Efficient Learning of Deep Networks from Decentralized Data, which will hencforth be refered to a s teh FedAvg paper

import torch
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
from tqdm import tqdm
import random

from networks import FashionNet, TwoNN
model_class = FashionNet

print('starting')

criterion = torch.nn.CrossEntropyLoss()

# this size of batches during testing
TEST_BATCH_SIZE = 32
# the size of batches during client updates
B = 10
# number of iid data samples assigned to each client
n_k = 6000
# number of clients
K = 10
# the number of passes over their data set that each client makes
E = 1
# fraction of clients queried per communication round
C = 1
# the learning rate
eta = 0.0001

torch.cuda.empty_cache()
device = 'cuda:0'
# device = 'cpu'

print('importing data')
raw_data = mnist.load_data()

x_train_raw = raw_data[0][0]
y_train_raw = raw_data[0][1]
x_test_raw = raw_data[1][0]
y_test_raw = raw_data[1][1]

x_train = torch.tensor(x_train_raw, dtype=torch.float32).reshape([-1, 1, 28, 28])/255
x_test = torch.tensor(x_test_raw, dtype=torch.float32).reshape([-1, 1, 28, 28])/255

y_train = torch.tensor(y_train_raw, dtype=torch.long)
y_test = torch.tensor(y_test_raw, dtype=torch.long)

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

def average_parameters(param_list):

	num_clients = len(param_list)
	avg_params = []

	for i, params in enumerate(param_list):

		if (i == 0):
			for p in params:
				avg_params.append(p.clone()/num_clients)

		else:
			for j, p in enumerate(params):
				avg_params[j].data += p.data/num_clients

	return avg_params

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

print('instantiating networks')
central_model = model_class().to(device)

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

num_global_updates = 500
for i in range(num_global_updates):
	print('--------------------------------------------------------------------------------')
	print('global training cycle', i+1)

	# randomly selecting clients
	client_indices = random.sample(list(range(K)), k=round(C*K))
	S = [clients[i] for i in client_indices]

	print('performing '+str(len(S))+' client updates with data allocations of sizes: '+str([len(k['data_indices']) for k in S]))
	for k in S:
		# sends parameters to each selected client
		set_parameters(k['net'], central_model.parameters())

		# shuffles data
		random.shuffle(k['data_indices'])

		# client updates these parameters on their local dataset
		client_update(k)

	# print('assessing client parameters')
	# print('loss | acc %')
	# for k in S:
	# 	loss, acc = val_evaluation(k['net'])
	# 	print(round(loss, 2), '|', round(100*acc, 1))

	print('aggregating')
	param_list = [k['net'].parameters() for k in S]

	weights = [len(k['data_indices']) for k in S]
	weights = [w/sum(weights) for w in weights]
	avg_params = weighted_average_parameters(param_list, weights)

	# setting central model to the average of the parameters of the clients' models
	set_parameters(central_model, avg_params)

	print('assessing aggregated parameters')
	loss, acc = val_evaluation(central_model)
	print('loss | acc: ', round(loss, 2), '|', round(100*acc, 1))

	# -------------------------------------------------------------------------------------------------------

	# print('we now train the aggregated parameters')

	# a = {
	# 	'net': central_model,
	# 	'optimizer': torch.optim.Adam([{'params': central_model.parameters()}], lr=0.0001),
	# 	'data_indices': torch.randperm(x_train.shape[0])[0: 10000]
	# }

	# for i in range(10):
	# 	client_update(a)
	# 	loss, acc = val_evaluation(central_model)
	# 	print(round(loss, 2), '|', round(100*acc, 1))