from tasks import tasks
from states import state_dicts
from itertools import product
from config import config
from utils import get_parameter_server
from tqdm import tqdm
from states import get_state, set_state, allowed_states, stressor_thread, training_stressor

import time
import pickle

device = config.training_device
task_names = tasks.keys()
# the number of times each learner will download the model while benchmarking
benchmark_downloads = 5
# the number of shards each learner will train on while benchmarking
benchmark_shards = 20
BATCH_SIZE = 32
dst_file = './benchmark_scores.pickle'

def run_benchmarks():
	benchmark_scores = {}

	for task_name, state_name in product(task_names, allowed_states):
		set_state(state_name)

		# wait a second for the new stressor to run
		time.sleep(1)

		# run benchmark
		benchmark_scores[(task_name, state_name)] = benchmark(task_name, benchmark_downloads, benchmark_shards)

	return benchmark_scores

download_stressor_size = 900
training_stressor_size = 900

def benchmark(task_name='mnist_ffn', num_downloads=1, num_shards=3):
	print(f'running benchmark for {task_name} in state: {get_state()}')

	global device
	task_description = tasks[task_name]

	# given the task, get the neural architecture and data shape from the 
	
	ps = get_parameter_server()
	state = get_state()
	stressor_handle = None

	# downloads parameters, times it, and takes the average
	print('downloading parameters')
	start_time = time.time()

	for i in range(num_downloads):

		if (state == 'downloading'):
			stressor_handle = ps.rpcs.dummy_download.async_call((download_stressor_size, download_stressor_size), {})
			stressor_handle.join()

		call_handle = ps.rpcs.get_parameters.async_call((task_name, ), {})
		parameters = call_handle.join()

	print('finished downloading parameters')
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

	if (state == 'downloading'):
			stressor_handle = ps.rpcs.dummy_download.async_call((download_stressor_size, download_stressor_size), {})

	x_shards, y_shards = ps.rpcs.get_training_data.sync_call((task_name, num_shards, ), {})

	if (state == 'downloading'):
			stressor_handle.join()

	end_time = time.time()

	print('finished downloading data')
	data_time_spb = (end_time - start_time)/num_shards

	print('reshaping data')
	x_benchmark = x_shards.reshape([x_shards.shape[0]*x_shards.shape[1]]+task_description['data_shape'])
	y_benchmark = y_shards.reshape([y_shards.shape[0]*y_shards.shape[1]])

	num_batches = x_benchmark.shape[0] // BATCH_SIZE

	# we now train the network on this random data and time how long it takes
	print('training')
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

		if (state == 'training'):
			training_stressor(training_stressor_size)

	print('finished training')
	end_time = time.time()

	# calcuating the training rate of the worker in batches per second
	training_rate_bps = num_shards/(end_time - start_time)

	del x_benchmark, y_benchmark

	return training_rate_bps, data_time_spb, param_time_spb

if (__name__ == '__main__'):
	# stressor_thread.start()

	print('warming up')
	for tn in task_names:
		print(tn)
		benchmark(task_name=tn)

	benchmark_scores = run_benchmarks()

	print('      | training rate bps | data time spb | param_time spb')

	for task_name, state_name in product(task_names, allowed_states):
		print(f'{task_name} in {state_name}: {benchmark_scores[(task_name, state_name)]}')

	print(f'writing to {dst_file}')

	with open(dst_file, 'wb') as f:
		f.write(pickle.dumps(benchmark_scores))