import axon
import signal
import time
import torch
import pickle
import random
import sys

sys.path.append('..')

from tqdm import tqdm
from threading import Thread, Lock

from config import config
from tasks import tasks
from states import state_dicts, state, stressor_thread, set_state
from utils import set_parameters, get_parameter_server
from benchmark import dst_file

# the price this worker charges per data sample
price = 0.1
num_test_shards = 20
BATCH_SIZE = 32
state_names = [state_name for state_name in state_dicts]

task_names = [task_name for task_name in tasks]
task_name = config.default_task_name
num_shards = config.delta
num_iters = 1
device = config.training_device

# holds the learner's benchmark scores once they're read from disk
benchmark_scores = {}

# makes set_state an RPC so the orchestrator can set the state of this worker
axon.worker.rpc()(set_state)

@axon.worker.rpc()
def get_benchmark_scores():
	global benchmark_scores
	return benchmark_scores

@axon.worker.rpc()
def set_price(new_price):
	global price
	price = new_price

@axon.worker.rpc()
def get_price():
	global price
	return price

@axon.worker.rpc()
def set_training_regime(incoming_task_name=config.default_task_name, incomming_num_shards=config.delta, incomming_num_iters=1):
	global task_name, num_shards, num_iters
	task_name = incoming_task_name
	num_shards = incomming_num_shards
	num_iters = incomming_num_iters

@axon.worker.rpc()
def local_update():
	print('performing local update routine')

	global device, task_name, num_shards, num_iters, state

	task_description = tasks[task_name]
	ps = get_parameter_server()

	print(f'training {task_name} in state: {state}')
	print('downloading parameters')
	parameters = ps.rpcs.get_parameters.sync_call((task_name, ), {})

	print('instantiating neural network')
	net = task_description['network_architecture']()
	set_parameters(net, parameters)

	# moves neural net to GPU, if one is available
	net.to(device)

	# creates loss and optimizer objects
	optimizer = task_description['optimizer'](net)
	criterion = task_description['loss']

	# downloading num_shards data samples
	print('downloading data')
	x_shards, y_shards = ps.rpcs.get_training_data.sync_call((task_name, num_shards, ), {})

	print('reshaping data')
	x_train = x_shards.reshape([x_shards.shape[0]*x_shards.shape[1]]+task_description['data_shape'])/255.0
	y_train = y_shards.reshape([y_shards.shape[0]*y_shards.shape[1]])
	
	num_batches = x_train.shape[0] // BATCH_SIZE

	# we now train the network on this random data and time how long it takes

	# training the network on random data
	print('training')
	for i in range(num_iters):
		print(f'iteration {i+1} of {num_iters}')
		for batch_number in tqdm(range(num_batches)):
			# getting batch
			x_batch = x_train[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(device)
			y_batch = y_train[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(device)

			# getting network's loss on batch
			y_hat = net(x_batch)

			loss = criterion(y_hat, y_batch)

			# updating parameters
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

	# marshalls the neural net's parameters
	param_update = [p.to('cpu') for p in list(net.parameters())]

	print('uploading parameters to PS')
	ps.rpcs.submit_update.sync_call((task_name, param_update, num_shards, ), {})

	# clearing data and parameters to save memory
	del x_shards, y_shards, x_train, y_train, parameters, param_update

def shutdown_handler(a, b):
	axon.discovery.sign_out(ip=config.notice_board_ip)
	exit()

if (__name__ == '__main__'):

	# registers sign out on sigint
	signal.signal(signal.SIGINT, shutdown_handler)

	# reads benchmark scores from disk
	with open(dst_file, 'rb') as f:
		buffer = f.read()
		benchmark_scores = pickle.loads(buffer)

	# starts stressor thread, this runs stressors that put the worker in different states
	stressor_thread.start()

	# sign into notice board, this is so that clients can discover this worker
	axon.discovery.sign_in(ip=config.notice_board_ip)

	# starts worker
	print('starting worker')
	axon.worker.init()