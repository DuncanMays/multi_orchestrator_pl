from tasks import tasks
from states import state_dicts
from itertools import product
from config import config
from utils import get_parameter_server
from tqdm import tqdm
from states import allowed_states, stressor_thread

import time
import pickle

device = config.training_device
task_names = [task_name for task_name in tasks]
# the number of times each learner will download the model while benchmarking
benchmark_downloads = 5
# the number of shards each learner will train on while benchmarking
benchmark_shards = 10
BATCH_SIZE = 32
state = 'idle'
dst_file = './benchmark_scores.pickle'

def run_benchmarks():
	benchmark_scores = {}

	for task_name, state_name in product(task_names, allowed_states):
		set_state(state_name)
		benchmark_scores[(task_name, state_name)] = benchmark(task_name, benchmark_downloads, benchmark_shards)

	return benchmark_scores

def set_state(new_state='idle'):
	global state

	if new_state in allowed_states: 
		state = new_state

def benchmark(task_name='minst_ffn', num_downloads=1, num_shards=3):
	print(f'running benchmark for {task_name} in state: {state}')

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
	# set_parameters(net, parameters)

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
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

	end_time = time.time()

	# calcuating the training rate of the worker in batches per second
	training_rate_bps = num_shards/(end_time - start_time)

	del x_benchmark, y_benchmark

	return training_rate_bps, data_time_spb, param_time_spb

if (__name__ == '__main__'):
	stressor_thread.start()

	benchmark_scores = run_benchmarks()

	for task_name, state_name in product(task_names, allowed_states):
		print(f'benchmark scores for {task_name} in {state_name} are: {benchmark_scores[(task_name, state_name)]}')

	print(f'writing to {dst_file}')

	with open(dst_file, 'wb') as f:
		f.write(pickle.dumps(benchmark_scores))