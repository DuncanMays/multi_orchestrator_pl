from tasks import tasks, global_budget
import axon
import asyncio
import random
import json
import time
import torch
from sys import argv as arg_list
from os.path import join as path_join
from itertools import product

from states import state_dicts
from config import config
from data_allocation import MED, MMTT
from worker_composite import WorkerComposite
from utils import get_parameter_server
from types import SimpleNamespace

arg_list = arg_list[1:]
arg_dict = {}
for index, arg in enumerate(arg_list):
	if (arg[0] == '-'):
		arg_dict[arg[1:]] = arg_list[index+1]

args = SimpleNamespace(**arg_dict)

# the number of times each learner will download the model while benchmarking
benchmark_downloads = 5
# the number of shards each learner will train on while benchmarking
benchmark_shards = 10
# the number of shards that is used to assess the testing loss and accuracy of each neural net
testing_shards = 10
# the number of times the orchestrator will attempt the data allocation calculation before quiting
num_retries = 10

task_names = [task_name for task_name in tasks]
state_names = [state_name for state_name in state_dicts]
parameter_server = get_parameter_server()

default_arguements = SimpleNamespace(
	data_allocation_regime = 'MED',
	state_distribution = 'uncertain',
	experiment_name = 'default',
	trial_index = '-1',
	num_learners = str(9),
	heat = str(0.5),
	num_tasks = len(task_names),
	deadline_adjust = 0
)

def overwrite(a, b):
	new_obj = {}
	keys = list(b.__dict__.keys())

	for k in keys:
		try:
			new_obj[k] = a.__dict__[k]
		except(KeyError):
			new_obj[k] = b.__dict__[k]

	return SimpleNamespace(**new_obj)

args = overwrite(args, default_arguements)

# this is where results will be recorded
result_folder = './results/Nov_1_meeting/'

result_file = args.data_allocation_regime+'_'+args.state_distribution+'_'+str(args.deadline_adjust)+'_'+args.experiment_name+'_'+args.trial_index+'.json'
result_file_path = path_join(result_folder, result_file)

# this is the dict where metrics will be recorded for each training run
# each task_name corresponds to a list of data points, each data point representing a global update cycle
# each data point holds: the time that the cycle took, and the loss and acc
training_metrics = {task_name: [] for task_name in task_names}

# this is the object that will be saved for each training run
results = {
	'training_metrics': training_metrics,
	'cost': 0,
	'time_prediction': 0,
}

def record_results(file_path):
	try:
		with open(file_path, 'w') as f:
			json_str = json.dumps(results)
			f.write(json_str)

	except(FileNotFoundError):
		print(f'file not found at {file_path}, recording data in backup.json')
		with open('./backup.json', 'w') as f:
			json_str = json.dumps(results)
			f.write(json_str)

async def training_routine(task_name, cluster_handle, num_training_cycles):

	# the number of workers on the given task
	num_learners_on_task = len(cluster_handle.children)

	# assessing the accuracy and loss before training
	loss, acc  = await parameter_server.rpcs.assess_parameters(task_name, testing_shards)
	print(f'the loss and accuracy for {task_name} was: {acc, loss} prior to training')

	# each data point represents a single global update iteration
	# times holds a list of the amount of time it was projected that each update should have taken on each worker
	# loss and acc hold the loss and accuracy of the aggregated neural network
	# cost holds the total price of assignment
	data_point = {'times': [], 'acc': acc, 'loss': loss, 'max_div': 0, 'mean_div': 0}

	training_metrics[task_name].append(data_point)

	for i in range(num_training_cycles):

		# randomly sets the state in each of the workers, according to each worker's state_distribution
		await cluster_handle.rpcs.set_state()

		# performs the local training routine on each worker on the task
		timings = await cluster_handle.rpcs.local_update()
		# local_update returns a 2-tuple of the amount of time it took to complete the update and then the extra time it should have taken if the update wasn't cut short by the deadline
		timings = [t[0] for t in timings]

		# aggregating parameters
		max_div, mean_div = await parameter_server.rpcs.aggregate_parameters(task_name)

		# assessing parameters
		loss, acc = await parameter_server.rpcs.assess_parameters(task_name, testing_shards)
		print(f'cycle completed on {task_name}, loss and accuracy: {loss, acc}')

		# recording results
		data_point = {'times': timings, 'acc': acc, 'loss': loss, 'max_div': max_div, 'mean_div': mean_div}
		training_metrics[task_name].append(data_point)

# the ideal state distribution
get_idle_dist = lambda : [float(sum([i == 0])) for i in range(len(state_names))]

# a one-hot vector
get_one_hot = lambda i : [float(sum([j == i])) for j in range(len(state_names))]

# a random vector
get_rand_vec = lambda : [random.uniform(0, 1) for j in range(len(state_names))]

# samples a normal distribution to obtain worker prices
get_random_price = lambda : min(max(random.gauss(config.worker_price_mean, config.worker_price_variance), config.worker_price_min), config.worker_price_max)

def create_state_distribution(heat):
	# this is the distribution that controls how workers change states
	d = torch.softmax(torch.tensor(get_one_hot(random.randint(0, len(state_names)-1)))/heat, dim=0).tolist()
	return d

# this function sets all the values that vary in between experiments, given the same set of workers between experiments
# the two peices of information that change are the worker state distributions and the worker prices, both of which are returned by this function
def initialize_parameters(num_learners):
	worker_prices = [get_random_price() for _ in range(num_learners)]

	state_distributions = None

	if args.state_distribution == 'ideal':
		# state distributions are ideal
		state_distributions = [get_idle_dist() for i in range(num_learners)]

	elif args.state_distribution == 'uncertain':
		# randomly setting the state distributions of each worker, based on heat parameter
		state_distributions = [create_state_distribution(float(args.heat)) for _ in range(num_learners)]

	elif args.state_distribution == 'static':
		# sampling from randomly set state distributions
		prime_state_dists = [create_state_distribution(float(args.heat)) for _ in range(num_learners)]
		# the state distribution that's sent to learners will be a one-hot vector representing the state selected from the distribution initialized above
		# this will mean the worker's state is constant, or static
		states = [random.choices(range(len(state_names)), weights=state_dist, k=1).pop() for state_dist in prime_state_dists]
		state_distributions = [get_one_hot(s) for s in states]

	else:
		raise BaseException('unknown state_distribution: ', args.state_distribution)

	return worker_prices, state_distributions

# returns a boolean value of weather or not there's an active worker at the given IP
async def test_ip(ip):
	try:
		handle = axon.client.RemoteWorker(ip)
		await handle.rpcs.get_benchmark_scores()
		return True

	except(BaseException):
		return False

async def get_active_learners(learner_ips):
	test_coros = []

	for ip in learner_ips:
		test_coros.append(test_ip(ip))

	test_results = await asyncio.gather(*test_coros)

	active_ips = []
	for i, ip in enumerate(learner_ips):
		if test_results[i]:
			active_ips.append(ip)

	return active_ips

def multiple_mnist_cnn(learner_scores):
	new_learner_scores = {}

	for i, state in product(range(5), state_names):
		new_learner_scores[('mnist_cnn_'+str(i), state)] = learner_scores[('mnist_cnn', state)]

	return new_learner_scores

async def main():
	# resets the parameter server from the last training run
	clear_promises = []
	for task_name in task_names:
		clear_promises.append(parameter_server.rpcs.clear_params(task_name))

	await asyncio.gather(*clear_promises)

	# get the IP addresses of each learner from the notice board
	learner_ips = axon.discovery.get_ips(ip=config.notice_board_ip)
	# the number of learners used in this trial, from command line arguement
	num_learners = int(args.num_learners)

	# we now filter out the IPs that are unresponsive
	learner_ips = await get_active_learners(learner_ips)

	if (len(learner_ips) < num_learners):
		raise(BaseException(f'{num_learners} learners requested but only {len(learner_ips)} are available'))

	# sorting learner_ips in order of the last digit
	# learner_ip_sort_key = lambda ip : int(ip.split('.')[-1])
	# learner_ips.sort(key=learner_ip_sort_key)

	# shuffling learner_ips
	random.shuffle(learner_ips)

	# selecting only the specified number of learners
	learner_ips = learner_ips[0:num_learners]

	print('learner_ips:', learner_ips)

	# creates worker handles
	learner_handles = [axon.client.RemoteWorker(ip) for ip in learner_ips]

	# This worker composite sends commands to the whole cluster in a single line
	global_cluster = WorkerComposite(learner_handles)

	# benchmark scores is a list of a maps from (task, state) hashes to (training_rate_bps, data_time_spb, param_time_spb) tuples, each index being a learner
	benchmark_scores = await global_cluster.rpcs.get_benchmark_scores()

	# this line creates a new benchmark object that only gives the scores for mnist_cnn for multiple identical tasks
	# benchmark_scores = [multiple_mnist_cnn(learner_scores) for learner_scores in benchmark_scores]

	# getting the random parameters for this trials
	worker_prices, state_distributions = initialize_parameters(num_learners)

	# the variables set by the optimization formulations
	association, allocation, iterations, time_predictions = None, None, None, None

	for task_name in task_names:
		 tasks[task_name]['deadline'] = tasks[task_name]['deadline'] + float(args.deadline_adjust)
	
	# try allocating a number of times, should mitigate 0 solution counts
	for i in range(num_retries):
		try:
			if (args.data_allocation_regime == 'MED'):
				association, allocation, time_predictions = MED(benchmark_scores, worker_prices, state_distributions, tasks)

			elif (args.data_allocation_regime == 'MMTT'):
				association, allocation, time_predictions = MMTT(benchmark_scores, worker_prices, state_distributions, tasks)

			else:
				raise(BaseException('unrecognized data_allocation_regime'))

			break

		except(AttributeError):
			# if allocation fails, try another set of parameters
			worker_prices, state_distributions = initialize_parameters(num_learners)

	iterations = [tasks[association[i]]['num_training_iters'] for i in range(num_learners)]

	print('-------------------- ', i)

	print(association)
	print(allocation)
	print(worker_prices)
	print(time_predictions)

	# if the optimization calculation fails, there's no need to proceed past this point
	if (association == None):
		exit()

	cost = sum([a*p for (a, p) in zip(allocation, worker_prices)])
	results['cost'] = cost

	# after allocation calculation confirms the state distributions are viable, transmit them to workers so they can manage their state dynamics
	state_distributions = [(s, ) for s in state_distributions]
	await global_cluster.rpcs.set_state_distribution(state_distributions)

	# tells the learners which task they're associated with and how much data to use while training

	# incoming_task_name=config.default_task_name,
	# incomming_num_shards=config.delta,
	# incomming_num_iters=1,
	# check_deadline_every=10,
	# incoming_deadline=config.default_deadline

	task_settings = list(zip(association, allocation, iterations, [10 for _ in range(num_learners)], [tasks[task_name]['deadline'] for task_name in association]))
	await global_cluster.rpcs.set_training_regime(task_settings)

	# we now create composite objects out of the workers assigned to each task, since the tasks can be processed independantly of one another
	# this holds lists of worker handles, split by the task the worker is assigned, all workers in a list work on the same task
	task_piles = {name: [] for name in task_names}
	# this is a parallel dictionary that holds the prediction made for how long each worker will train for
	prediction_piles = {name: [] for name in task_names}

	for i in range(num_learners):
		# the task the learner is assigned
		task_name = association[i]
		# adds that learner to the pile associated with that task
		task_piles[task_name].append(learner_handles[i])
		prediction_piles[task_name].append(time_predictions[i])

	# clusters of workers all working on the same task, indexed by the name of the task
	task_clusters = {task_name: WorkerComposite(pile) for task_name, pile in task_piles.items()}

	results['time_prediction'] = prediction_piles

	training_promises = []
	for task_name in task_clusters:
		training_promises.append(training_routine(task_name, task_clusters[task_name],  5))

	# deploys training routines in parallel
	await asyncio.gather(*training_promises)

	# records results to a file
	print(result_file_path)
	record_results(result_file_path)

if (__name__ == '__main__'):
	asyncio.run(main())
