import axon
import signal
import time
import torch
from tqdm import tqdm

from config import config
from tasks import tasks

# the price this worker charges per data sample
price = 0.1

BATCH_SIZE = 32

device = 'cpu'
if torch.cuda.is_available():
	device = 'cuda:0'

parameter_server = None
def get_parameter_server():
	global parameter_server

	if (parameter_server == None):
		parameter_server = axon.client.RemoteWorker(config.parameter_server_ip)

	return parameter_server

# sets the parameters of a neural net
def set_parameters(net, params):
	current_params = list(net.parameters())
	for i, p in enumerate(params): 
		current_params[i].data = p.data.clone()

@axon.worker.rpc()
def get_price():
	global price
	return price

# rpc that runs benchmark
@axon.worker.rpc()
def benchmark(task_name='minst_ffn', num_downloads=1, num_batches=100):
	print('running benchmark!')

	global device

	# given the task, get the neural architecture and data shape from the 
	
	ps = get_parameter_server()

	# downloads parameters, times it, and takes the average

	# sets parameters
	download_times = []
	for i in range(num_downloads):
		start_time = time.time()
		call_handle = ps.rpcs.get_parameters.async_call((task_name, ), {})
		parameters = call_handle.join()
		end_time = time.time()

		download_times.append(end_time - start_time)

	avg_time = sum(download_times)/num_downloads

	task_description = tasks[task_name]

	net = task_description['network_architecture']()

	set_parameters(net, parameters)

	# moves neural net to GPU, if one is available
	net.to(device)

	# creates loss and optimizer objects
	optimizer = task_description['optimizer'](net)
	criterion = task_description['loss']

	# creating random data
	x_benchmark = torch.randn([BATCH_SIZE*num_batches]+task_description['data_shape'], dtype=torch.float32)
	y_benchmark = torch.ones([BATCH_SIZE*num_batches], dtype=torch.long)

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

	# calcuating the training rate of the worker
	batches_per_second = num_batches/(end_time - start_time)

	return batches_per_second, avg_time

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

