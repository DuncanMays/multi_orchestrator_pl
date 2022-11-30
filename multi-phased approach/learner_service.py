import axon
import signal
import time
import torch
import pickle
import random
import sys
from args import args

sys.path.append('..')

from tqdm import tqdm
from types import SimpleNamespace
from config import config
from tasks import tasks
from states import training_stressor
from utils import set_parameters, get_parameter_server
from benchmark import dst_file

BATCH_SIZE = 32
training_device = config.training_device
allowed_states = ['idle', 'downloading', 'training']

class Learner():

	def __init__(self, name, parameter_server):

		self.price = 0.1
		self.task_name = config.default_task_name
		self.num_shards = config.delta
		self.num_iters = 1
		self.check_deadline_interval = 10
		self.deadline = config.default_deadline
		self.download_stressor_size = 900
		self.training_stressor_size = 900
		self.state = 'idle'
		self.state_distribution = [1/len(allowed_states) for i in allowed_states]
		self.halting = True
		self.name = name
		self.ps = parameter_server

		# reads benchmark scores from disk
		self.benchmark_scores = {}
		with open(dst_file, 'rb') as f:
			buffer = f.read()
			self.benchmark_scores = pickle.loads(buffer)

	def set_state(self, new_state=None):
		if (new_state == None):
			new_state = random.choices(allowed_states, self.state_distribution).pop()
			self.state = new_state

		elif new_state in allowed_states:
			self.state = new_state

		elif new_state not in allowed_states:
			raise(BaseException(f'invalid state setting: {new_state}'))

	def set_state_distribution(self, new_distribution):
		if (sum(new_distribution) >= 1.01) or (sum(new_distribution) <= 0.99):
			raise BaseException('invalid probability distribution')

		if (len(new_distribution) != len(allowed_states)):
			raise BaseException('number of indices in distribution list doesn\'t match number of states')

		self.state_distribution = new_distribution

	def get_benchmark_scores(self, ):
		return self.benchmark_scores

	def set_halting(self, new_halting):
		self.halting = new_halting

	def set_training_regime(self, 
			incoming_task_name=config.default_task_name,
			incomming_num_shards=config.delta,
			incomming_num_iters=1,
			incomming_deadline=config.default_deadline
		):

		print(self.name, 'set training regime')
		print(incoming_task_name)
		
		self.task_name = incoming_task_name
		self.num_shards = incomming_num_shards
		self.num_iters = incomming_num_iters
		self.deadline = incomming_deadline

	def get_data_allocated(self, ):
		return self.num_shards

	def local_update(self, ):
		global BATCH_SIZE

		print(self.name, 'performing local update routine')

		task_description = tasks[self.task_name]
		print(self.name, self.task_name)
		stressor_handle = None
		start_time = time.time()

		# returns True if the worker is past the deadline
		check_deadline = lambda : (time.time() - start_time) > self.deadline

		print(self.name, f'training {self.task_name} in state: {self.state}')
		print(self.name, 'downloading parameters')

		if (self.state == 'downloading'):
			stressor_handle = self.ps.rpcs.dummy_download.async_call((self.download_stressor_size, self.download_stressor_size), {})
			stressor_handle.join()

		parameters = self.ps.rpcs.get_parameters.sync_call((self.task_name, ), {})

		print(self.name, 'instantiating neural network')
		net = task_description['network_architecture']()
		set_parameters(net, parameters)

		# moves neural net to GPU, if one is available
		net.to(training_device)

		# creates loss and optimizer objects
		optimizer = task_description['optimizer'](net)
		criterion = task_description['loss']

		# downloading num_shards data samples
		print(self.name, 'downloading data')

		if (self.state == 'downloading'):
			stressor_handle = self.ps.rpcs.dummy_download.async_call((self.download_stressor_size, self.download_stressor_size), {})
			stressor_handle = self.ps.rpcs.dummy_download.async_call((self.download_stressor_size, self.download_stressor_size), {})

		x_shards, y_shards = self.ps.rpcs.get_training_data.sync_call((self.task_name, self.num_shards, ), {})

		if (self.state == 'downloading'):
			stressor_handle.join()

		print(self.name, 'reshaping data')
		x_train = x_shards.reshape([x_shards.shape[0]*x_shards.shape[1]]+task_description['data_shape'])/255.0
		y_train = y_shards.reshape([y_shards.shape[0]*y_shards.shape[1]])
		
		num_batches = x_train.shape[0] // BATCH_SIZE
		# ratio of training loop that has been completed in the event that the training is cut short by the deadline
		batches_completed = 0
		training_start_time = time.time()
		# boolean holding True if the worker exceeds its deadline while training
		halted = False

		# training the network
		print(self.name, 'training, deadline is ', self.deadline)
		for i in range(self.num_iters):
			print(self.name, f'iteration {i+1} of {self.num_iters}')
			for batch_number in tqdm(range(num_batches)):
				# getting batch
				x_batch = x_train[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(training_device)
				y_batch = y_train[batch_number*BATCH_SIZE: (batch_number+1)*BATCH_SIZE].to(training_device)

				# getting network's loss on batch
				y_hat = net(x_batch)

				loss = criterion(y_hat, y_batch)

				# updating parameters
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

				if (self.state == 'training'):
					training_stressor(self.training_stressor_size)

				if (batch_number%self.check_deadline_interval == 0) and self.halting:
					if check_deadline():
						# the number of batches completed in this epoch
						batches_completed += batch_number + 1
						# raises flag indicating halt
						halted = True
						# end the loop over batches
						break

			if halted:
				# end the training loop over epochs
				break
			else:
				# else increment completed batches counter
				batches_completed += num_batches

		# the amount of time spent training
		training_time = time.time() - training_start_time

		# marshalls the neural net's parameters
		param_update = [p.to('cpu') for p in list(net.parameters())]

		print(self.name, 'uploading parameters to PS')
		if (self.state == 'downloading'):
			stressor_handle = self.ps.rpcs.dummy_download.async_call((self.download_stressor_size, self.download_stressor_size), {})
			stressor_handle = self.ps.rpcs.dummy_download.async_call((self.download_stressor_size, self.download_stressor_size), {})

		self.ps.rpcs.submit_update.sync_call((self.task_name, param_update, batches_completed, ), {})

		if (self.state == 'downloading'):
			stressor_handle.join()

		# calculates the time remaining
		total_batches = self.num_iters*num_batches
		completed_ratio = batches_completed/total_batches
		if (completed_ratio != 1):
			time_remaining = training_time*(1/completed_ratio - 1)
		else:
			time_remaining = 0.0

		total_time = time.time() - start_time

		return total_time, min(100, max(time_remaining, 0))

# starts the service
def main():

	# the parameter server
	ps = get_parameter_server()
	# the notice_board
	nb = axon.client.ServiceStub(config.notice_board_ip, endpoint_prefix='notice_board')

	num_learners = None
	try:
		num_learners = int(args.n)
	except(AttributeError):
		num_learners = 1

	sign_in_threads = []

	for i in range(num_learners):
		learner_name = 'l_'+str(i)
		learner = Learner(learner_name, ps)
		axon.worker.ServiceNode(learner, learner_name)
		sign_in_threads.append(nb.sign_in.___call___.async_call((axon.utils.get_self_ip(), learner_name), {}))

	print('signing in to notice board')
	[t.join() for t in sign_in_threads]

	# this function lets the notice board know that these learners are no longer available
	def shutdown_handler(a, b):
		sign_out_threads = []

		for i in range(num_learners):
			learner_name = 'l_'+str(i)
			sign_out_threads.append(nb.sign_out.___call___.async_call((axon.utils.get_self_ip(), learner_name), {}))

		try:
			[t.join() for t in sign_out_threads]
		except(BaseException):
			pass

		exit()

	# signs out on siging
	signal.signal(signal.SIGINT, shutdown_handler)

	print(f'starting {num_learners} learners')
	axon.worker.init()

if (__name__ == '__main__'):
	main()
