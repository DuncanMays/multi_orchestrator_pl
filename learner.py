import axon
import signal
import time
import torch

from config import config

# rpc that runs benchmark
@axon.worker.rpc()
def benchmark(task, num_batches):
	print('running benchmark!')

	# given the task, get the neural architecture and data shape
	global net, criterion, device

	net.to(device)
	optimizer = get_optimizer(net)

	# creating random data
	x_benchmark = torch.randn([BATCH_SIZE*num_batches, 784], dtype=torch.float32)
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

	return batches_per_second

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
	time.sleep(60)

	axon.worker.init()