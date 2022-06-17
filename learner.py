import axon
import signal
import time
import torch
from tqdm import tqdm
from threading import Thread, Lock
from itertools import product

from config import config
from tasks import tasks
from states import state_dicts
from utils import set_parameters, get_parameter_server

# the price this worker charges per data sample
price = 0.1
num_test_shards = 20
BATCH_SIZE = 32

task_names = [task_name for task_name in tasks]
task_name = config.default_task_name
num_shards = config.delta

allowed_states = [state_name for state_name in state_dicts]
state_lock = Lock()
state = allowed_states[0]

# the number of times each learner will download the model while benchmarking
benchmark_downloads = 5
# the number of shards each learner will train on while benchmarking
benchmark_shards = 10

device = config.training_device

parameter_server = get_parameter_server()

# ------------------------------------------------------------------------------------------

def top_level_stressor():
	print('top level stressor has begun')

	# runs a stressor functions once a second based on the state
	while (True):
		time.sleep(1)

		with state_lock:
			stressor_fn = state_dicts[state]['stressor_fn']
			stressor_params = state_dicts[state]['params']

		stressor_fn(*stressor_params)

# this thread runs stressor functions that utilize a certain compute resource to change the learner's compute characteristics
stressor_thread = Thread(target=top_level_stressor)
stressor_thread.daemon = True
stressor_thread.start()

# ------------------------------------------------------------------------------------------

benchmark_scores = {}

@axon.worker.rpc()
def startup():
	global benchmark_scores

	# stores benchmarking scores in each state
	benchmark_scores = {}
	for task_name, state_name in product(task_names, allowed_states):
		set_state(state_name)
		task_state_hash = (task_names.index(task_name), allowed_states.index(state_name))
		benchmark_scores[task_state_hash] = benchmark(task_name, benchmark_downloads, benchmark_shards)

@axon.worker.rpc()
def set_state(new_state='idle'):
	global state

	if new_state in allowed_states: 
		with state_lock:
			state = new_state
	
	else:
		Raise(BaseException('invalid state setting'))

@axon.worker.rpc()
def get_benchmark_scores():
	return benchmark_scores

@axon.worker.rpc()
def get_price():
	return price

# rpc that runs benchmark
@axon.worker.rpc()
def benchmark(task_name='minst_ffn', num_downloads=1, num_shards=3):
	print(f'running benchmark in state: {state}')

	global device
	task_description = tasks[task_name]

	# given the task, get the neural architecture and data shape from the 
	
	ps = get_parameter_server()

	# downloads parameters, times it, and takes the average
	start_time = time.time()
	for i in range(num_downloads):
		call_handle = ps.rpcs.get_parameters.async_call((task_name, ), {})
		parameters = call_handle.join()

	end_time = time.time()
	param_time_spb = (end_time - start_time)/num_downloads

	net = task_description['network_architecture']()
	set_parameters(net, parameters)

	# moves neural net to GPU, if one is available
	net.to(device)

	# creates loss and optimizer objects
	optimizer = task_description['optimizer'](net)
	criterion = task_description['loss']

	# downloading num_shards data samples
	print('downloading data')
	start_time = time.time()
	x_shards, y_shards = ps.rpcs.get_training_data.sync_call((task_name, num_shards, ), {})
	end_time = time.time()

	data_time_spb = (end_time - start_time)/num_shards

	print('reshaping data')
	x_benchmark = x_shards.reshape([x_shards.shape[0]*x_shards.shape[1]]+task_description['data_shape'])
	y_benchmark = y_shards.reshape([y_shards.shape[0]*y_shards.shape[1]])

	num_batches = x_benchmark.shape[0] // BATCH_SIZE

	# we now train the network on this random data and time how long it takes
	start_time = time.time()

	# training the network on the data we just downloaded
	for batch_number in tqdm(range(num_batches)):
		# getting batch
		x_batch = x_benchmark[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(device)
		y_batch = y_benchmark[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(device)

		# getting network's loss on batch
		y_hat = net(x_batch)

		loss = criterion(y_hat, y_batch)

		# updating parameters
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	end_time = time.time()

	# calcuating the training rate of the worker in batches per second
	training_rate_bps = num_shards/(end_time - start_time)

	del x_benchmark, y_benchmark

	return training_rate_bps, data_time_spb, param_time_spb

@axon.worker.rpc()
def set_training_regime(incoming_task_name=config.default_task_name, incomming_num_shards=config.delta):
	global task_name, num_shards

	task_name = incoming_task_name
	num_shards = incomming_num_shards

@axon.worker.rpc()
def local_update():
	print('performing local update routine')

	global device, task_name, num_shards

	task_description = tasks[task_name]
	ps = get_parameter_server()

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
	for batch_number in tqdm(range(num_batches)):
		# getting batch
		x_batch = x_train[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(device)
		y_batch = y_train[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(device)

		# getting network's loss on batch
		y_hat = net(x_batch)

		loss = criterion(y_hat, y_batch)

		# updating parameters
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# marshalls the neural net's parameters
	param_update = [p.to('cpu') for p in list(net.parameters())]

	print('uploading parameters to PS')
	ps.rpcs.submit_update.sync_call((task_name, param_update, num_shards, ), {})

def shutdown_handler(a, b):
	axon.discovery.sign_out(ip=config.notice_board_ip)
	exit()

if (__name__ == '__main__'):

	# registers sign out on sigint
	signal.signal(signal.SIGINT, shutdown_handler)

	# sign into notice board
	axon.discovery.sign_in(ip=config.notice_board_ip)

	# starts worker
	print('starting worker')
	axon.worker.init()