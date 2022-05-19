import axon
import signal
import time
import torch
from tqdm import tqdm

from config import config
from tasks import tasks
from utils import set_parameters

# the price this worker charges per data sample
price = 0.1
num_test_shards = 20
BATCH_SIZE = 32

task_name = config.default_task_name
num_shards = config.delta

device = 'cpu'
if torch.cuda.is_available():
	device = 'cuda:0'

parameter_server = None
def get_parameter_server():
	global parameter_server

	if (parameter_server == None):
		parameter_server = axon.client.RemoteWorker(config.parameter_server_ip)

	return parameter_server

# this thread runs stressor functions that utilize a certain compute resource to change the learner's compute characteristics
stressor_thread = Thread(target=top_level_stressor)
stressor_thread.daemon = True

def assess_parameters(net, num_shards):

	task_description = tasks[task_name]

	x_shards, y_shards = parameter_server.rpcs.get_testing_data.sync_call((task_name, num_shards, ), {})

	x_test = x_shards.reshape([x_shards.shape[0]*x_shards.shape[1]]+task_description['data_shape'])
	y_test = y_shards.reshape([y_shards.shape[0]*y_shards.shape[1]])

	num_test_batches = x_test.shape[0]//BATCH_SIZE

	loss = 0
	acc = 0

	net = net.to(device)
	criterion = task_description['loss']

	for batch_number in range(num_test_batches):
		x_batch = x_test[BATCH_SIZE*batch_number : BATCH_SIZE*(batch_number+1)].to(device)
		y_batch = y_test[BATCH_SIZE*batch_number : BATCH_SIZE*(batch_number+1)].to(device)

		y_hat = net.forward(x_batch)

		loss += criterion(y_hat, y_batch).item()
		acc += get_accuracy(y_hat, y_batch).item()
	
	# normalizing the loss and accuracy
	loss = loss/num_test_batches
	acc = acc/num_test_batches

	return loss, acc

# calculates the accuracy score of a prediction y_hat and the ground truth y
I = torch.eye(10,10)
def get_accuracy(y_hat, y):
	y_vec = torch.tensor([I[int(i)].tolist() for i in y]).to(device)
	dot = torch.dot(y_hat.flatten(), y_vec.flatten())
	return dot/torch.sum(y_hat)

@axon.worker.rpc()
def get_price():
	global price
	return price

# rpc that runs benchmark
@axon.worker.rpc()
def benchmark(task_name='minst_ffn', num_downloads=1, num_shards=10):
	print('running benchmark!')

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
	call_handle = ps.rpcs.get_training_data.async_call((task_name, num_shards, ), {})
	x_shards, y_shards = call_handle.join()
	end_time = time.time()

	data_time_spb = (end_time - start_time)/num_shards

	print('reshaping data')
	x_benchmark = x_shards.reshape([x_shards.shape[0]*x_shards.shape[1]]+task_description['data_shape'])
	y_benchmark = y_shards.reshape([y_shards.shape[0]*y_shards.shape[1]])

	num_batches = x_benchmark.shape[0] // BATCH_SIZE

	# we now train the network on this random data and time how long it takes
	start_time = time.time()

	# training the network on random data
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

	print('assessing performance, loss and acc:', assess_parameters(net, num_test_shards))

	# creates loss and optimizer objects
	optimizer = task_description['optimizer'](net)
	criterion = task_description['loss']

	# downloading num_shards data samples
	print('downloading data')
	x_shards, y_shards = ps.rpcs.get_training_data.sync_call((task_name, num_shards, ), {})

	print('reshaping data')
	x_train = x_shards.reshape([x_shards.shape[0]*x_shards.shape[1]]+task_description['data_shape'])/255.0
	y_train = y_shards.reshape([y_shards.shape[0]*y_shards.shape[1]])

	# print(x_train)
	# print(x_train.mean())
	# print(x_train[0])
	# print(x_train[0][0])
	# print(x_train.shape)

	# print(y_train)
	# print(y_train.shape)
	
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

	print('assessing performance, loss and acc:', assess_parameters(net, num_test_shards))

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