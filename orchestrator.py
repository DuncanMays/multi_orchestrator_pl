import axon
import asyncio
import random
import json
import time
import torch
from sys import argv as arg_list
from os.path import join as path_join

from tasks import tasks, global_budget
from states import state_dicts
from config import config
from data_allocation import EOL_prime, RSS, MED, MMTT
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

heat_param = 0.5

default_arguements = SimpleNamespace(
	data_allocation_regime = 'EOL',
	ideal_worker_state = 'False',
	experiment_name = 'default',
	trial_index = '-1',
	num_learners = str(len(task_names))
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
result_folder = './results/Sep_15_meeting'

state_dist_name = None

if args.ideal_worker_state == 'True':
	state_dist_name = 'ideal'
else:
	state_dist_name = 'uncertain'

result_file = args.data_allocation_regime+'_'+state_dist_name+'_'+args.num_learners+'_'+args.experiment_name+'_'+args.trial_index+'.json'
result_file_path = path_join(result_folder, result_file)

# this is the dict where metrics will be recorded for each training run
# each task_name corresponds to a list of data points, each data point representing a global update cycle
# each data point holds: the time that the cycle took, and the loss and acc
# data_point = {time, acc, loss, sat_ratio, cost}
training_metrics = {task_name: [] for task_name in task_names}

# this is the object that will be saved for each training run
results = {
	'training_metrics': training_metrics,
	'cost': 0,
	'EOL': 0,
}

def create_state_distribution(heat):
	# this is the distribution that controls how workers change states
	d = torch.softmax(torch.randint(0, 2, (len(state_names), ))/heat, dim=0).tolist()
	return d

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
	data_point = {'times': [], 'acc': acc, 'loss': loss}

	training_metrics[task_name].append(data_point)

	for i in range(num_training_cycles):

		# randomly sets the state in each of the workers, according to each worker's state_distribution
		await cluster_handle.rpcs.set_state()

		# performs the local training routine on each worker on the task
		timings = await cluster_handle.rpcs.local_update()
		# local_update returns a 2-tuple of the amount of time it took to complete the update and then the extra time it should have taken if the update wasn't cut short by the deadline
		timings = [sum(t) for t in timings]

		# aggregating parameters
		await parameter_server.rpcs.aggregate_parameters(task_name)

		# assessing parameters
		loss, acc = await parameter_server.rpcs.assess_parameters(task_name, testing_shards)
		print(f'cycle completed on {task_name}, loss and accuracy: {loss, acc}')

		# recording results
		data_point = {'times': timings, 'acc': acc, 'loss': loss}
		training_metrics[task_name].append(data_point)

# the ideal state distribution
get_idle_dist = lambda : [sum([i == 0]) for i in range(len(state_names))]

# samples a normal distribution to obtain worker prices
get_random_price = lambda : min(max(random.gauss(config.worker_price_mean, config.worker_price_variance), config.worker_price_min), config.worker_price_max)

# this function sets all the values that vary in between experiments, given the same set of workers between experiments
# the two peices of information that change are the worker state distributions and the worker prices, both of which are returned by this function
def initialize_parameters(num_learners):
	worker_prices = [get_random_price() for _ in range(num_learners)]

	state_distributions = None

	if args.ideal_worker_state == 'True':
		# state distributions are ideal
		state_distributions = [get_idle_dist() for i in range(num_learners)]

	else:
		# randomly setting the state distributions of each worker, based on heat parameter
		state_distributions = [create_state_distribution(heat_param) for _ in range(num_learners)]

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

	# shuffling learners
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

	# getting the random parameters for this trials
	worker_prices, state_distributions = initialize_parameters(num_learners)
	# print(state_distributions)

	# the variables set by the optimization formulations
	association, allocation, iterations, EOL = None, None, None, None
	
	# try allocating a number of times, should mitigate 0 solution counts
	for i in range(num_retries):
		try:
			if (args.data_allocation_regime == 'EOL'):
				association, allocation, iterations, EOL = EOL_prime(benchmark_scores, worker_prices, state_distributions)
				# association, allocation, iterations, EOL = EOL_minmax(benchmark_scores, worker_prices, state_distributions)

			elif (args.data_allocation_regime == 'TT'):
				association, allocation, iterations, EOL = RSS(benchmark_scores, worker_prices, state_distributions)

			elif (args.data_allocation_regime == 'MED'):
				association, allocation, iterations, EOL = MED(benchmark_scores, worker_prices, state_distributions)

			elif (args.data_allocation_regime == 'MMTT'):
				association, allocation, iterations, EOL = MMTT(benchmark_scores, worker_prices, state_distributions)

			else:
				raise(BaseException('unrecognized data_allocation_regime'))

			break

		except(AttributeError):
			# if allocation fails, try another set of parameters
			worker_prices, state_distributions = initialize_parameters(num_learners)

	print('-------------------- ', i)

	print(association)
	print(allocation)
	print(worker_prices)
	print(iterations)
	print(EOL)

	# if the optimization calculation fails, there's no need to proceed past this point
	if (association == None):
		exit()

	cost = sum([a*p for (a, p) in zip(allocation, worker_prices)])
	results['cost'] = cost
	results['EOL'] = EOL

	# after allocation calculation confirms the state distributions are viable, transmit them to workers so they can manage their state dynamics
	state_distributions = [(s, ) for s in state_distributions]
	await global_cluster.rpcs.set_state_distribution(state_distributions)

	# tells the learners which task they're associated with and how much data to use while training
	task_settings = list(zip(association, allocation, iterations))
	await global_cluster.rpcs.set_training_regime(task_settings)

	# we now create composite objects out of the workers assigned to each task, since the tasks can be processed independantly of one another
	# this holds lists of worker handles, split by the task the worker is assigned, all workers in a list work on the same task
	task_piles = {name: [] for name in task_names}

	for i in range(num_learners):
		# the task the learner is assigned
		task_name = association[i]
		# adds that learner to the pile associated with that task
		task_piles[task_name].append(learner_handles[i])

	# clusters of workers all working on the same task, indexed by the name of the task
	task_clusters = {task_name: WorkerComposite(pile) for task_name, pile in task_piles.items()}

	training_promises = []
	for task_name in task_clusters:
		training_promises.append(training_routine(task_name, task_clusters[task_name],  5))

	# deploys training routines in parallel
	await asyncio.gather(*training_promises)

	# records results to a file
	record_results(result_file_path)

if (__name__ == '__main__'):
	asyncio.run(main())
