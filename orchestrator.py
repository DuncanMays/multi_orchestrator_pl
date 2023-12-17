from tasks import tasks, global_budget
import axon
import asyncio
import random
import json
import time
import torch
from sys import argv as arg_list
from os.path import join as path_join
from os.path import exists as file_exists
from itertools import product

from states import state_dicts
from config import config
from data_allocation import MED, MMTT_uncertain, MMTT_ideal
from worker_composite import WorkerComposite
from utils import get_parameter_server
from types import SimpleNamespace

arg_list = arg_list[1:]
arg_dict = {}
for index, arg in enumerate(arg_list):
	if (arg[0] == '-'):
		arg_dict[arg[1:]] = arg_list[index+1]

args = SimpleNamespace(**arg_dict)

default_arguements = SimpleNamespace(
	data_allocation_regime = 'MED',
	state_distribution = 'uncertain',
	experiment_name = 'default',
	trial_index = '-1',
	num_learners = str(1),
	heat = float(0.5),
	num_tasks = 3,
	deadline_adjust = 0,
	new_states = False,
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

# the number of shards that is used to assess the testing loss and accuracy of each neural net
testing_shards = 10
# the number of Global Update Cycle required for each task 
num_GUC = 5

# *** this line is needed to vary the number of tasks, but should otherwise be commented
tasks = {key: tasks[key] for key in list(tasks.keys())[0: int(args.num_tasks)]}
print(list(tasks.keys()))

task_names = [task_name for task_name in tasks]
state_names = [state_name for state_name in state_dicts]
parameter_server = get_parameter_server()

# this is where results will be recorded
result_folder = './results/vary_tasks/'

result_file = args.data_allocation_regime+'_'+args.state_distribution+'_'+args.experiment_name+'_'+args.trial_index+'.json'
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

# returns a boolean value of weather or not there's an active worker at the given IP
async def test_ip(ip):
	try:
		handle = axon.client.get_RemoteWorker(ip)
		await handle.rpcs.get_benchmark_scores()
		return True

	except(BaseException):
		print('oh no!')
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

# this function takes learner benchmark scores and returns benchmark scores for mnist_cnn for as many tasks are required
# this is for experiments in varying the number of learners
def multiple_mnist_cnn(learner_scores):
	new_learner_scores = {}

	for i, state in product(range(int(args.num_tasks)), state_names):
		new_learner_scores[('mnist_cnn_'+str(i), state)] = learner_scores[('mnist_cnn', state)]

	return new_learner_scores

def get_learner_states():
	# creates a new state_distribution using the one-hote method
	state_distribution = create_state_distribution(args.heat)
	# samples num_GUC from the state distribution
	states = random.choices(state_names, state_distribution, k=num_GUC)

	return {
		'states': states,
		'state_distribution': state_distribution
	}

# returns a dict from IP addresses to the description of state of each learner, including their state distribution and states
async def get_states():
	num_learners = config.default_num_learners
	learner_state_file = './not_learner_states.json'
	learner_states = None

	if args.new_states or not file_exists(learner_state_file):
		print('resetting states')
		# If the learner_state_file doesn't exist, or the user has set a CLI option, we reinitialize the states
		num_learners = int(args.num_learners)

		# get the IP addresses of each learner from the notice board
		print('from: axon.discovery.get_ips')
		learner_ips = axon.discovery.get_ips(ip=config.notice_board_ip)
		print(learner_ips)

		# we now filter out the IPs that are unresponsive
		learner_ips = await get_active_learners(learner_ips)

		# if there's not enough learners active, raise
		if (len(learner_ips) < num_learners):
			raise(BaseException(f'{num_learners} learners requested but only {len(learner_ips)} are available'))

		# shuffling learner_ips
		random.shuffle(learner_ips)

		# selecting num_learner IPs
		learner_ips = learner_ips[0:num_learners]

		# set states for each learner in trial
		learner_states = {ip: get_learner_states() for ip in learner_ips}

		# save the states to file
		with open(learner_state_file, 'w') as f:
			f.write(json.dumps(learner_states))

	else:
		print('loading states')
		# if the file doesn't exist and CLI option isn't set

		# load old state and IPs
		learner_states = None
		with open(learner_state_file, 'r') as f:
			learner_states = json.loads(f.read())

		# check to make sure all required learners are active and responsive
		learner_ips = list(learner_states.keys())
		print(learner_ips)
		active_ips = await get_active_learners(learner_ips)
		if len(active_ips) < len(learner_ips):
			raise BaseException('some of the required learners are not responsive')

	return learner_states

# runs num_training_cycles on the learners allocated to the task task_name
async def training_routine(task_name, cluster_handle, num_training_cycles, cluster_states, allocation):

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

	# if the state distribution is uncertain, then the orchestrator only allocates once at the beginning
	if (args.state_distribution == 'uncertain'):
			print(f'setting data allocations on learners for task {task_name}')

			task_settings = list(zip([task_name for _ in range(num_learners_on_task)], 
				allocation,
				[tasks[task_name]['num_training_iters'] for _ in range(num_learners_on_task)], 
				[True for _ in range(num_learners_on_task)], 
				[tasks[task_name]['deadline'] for _ in range(num_learners_on_task)],
				))

			await cluster_handle.rpcs.set_training_regime(task_settings)

	for i in range(num_training_cycles):

		# if the state distribution is ideal, then the orchestrator allocates data every GUC
		if (args.state_distribution == 'ideal'):
			print(f'setting data allocations on learners for task {task_name}')

			task_settings = list(zip(
				[task_name for _ in range(num_learners_on_task)], 
				[a[i] for a in allocation],
				[tasks[task_name]['num_training_iters'] for _ in range(num_learners_on_task)], 
				[True for _ in range(num_learners_on_task)], 
				[tasks[task_name]['deadline'] for _ in range(num_learners_on_task)]))

			await cluster_handle.rpcs.set_training_regime(task_settings)


		# setting the states on each learner
		cluster_ips = [child.ip_addr for child in cluster_handle.children]
		state_settings = [(cluster_states[ip]['states'][i], ) for ip in cluster_ips]
		await cluster_handle.rpcs.set_state(state_settings)

		# performs the local training routine on each worker on the task
		timings = await cluster_handle.rpcs.local_update()
		# local_update returns a 2-tuple of the amount of time it took to complete the update and then the extra time it should have taken if the update wasn't cut short by the deadline
		timings = [t[0] for t in timings]

		# aggregating parameters
		p = await parameter_server.rpcs.aggregate_parameters(task_name)
		# print(p)
		num_batches, max_div, mean_div = p

		# assessing parameters
		loss, acc = await parameter_server.rpcs.assess_parameters(task_name, testing_shards)
		print(f'cycle completed on {task_name}, loss and accuracy: {loss, acc}')

		# recording results
		data_point = {'times': timings, 'acc': acc, 'loss': loss, 'max_div': max_div, 'mean_div': mean_div, 'num_batches': num_batches}
		training_metrics[task_name].append(data_point)

async def main():
	# resets the parameter server from the last training run
	clear_promises = []
	for task_name in task_names:
		clear_promises.append(parameter_server.rpcs.clear_params(task_name))

	await asyncio.gather(*clear_promises)

	# the states that the learners are to take throughout the duration of the trial
	learner_states = await get_states()

	# the keys of the dict states or the IP addresses of the learners involved
	learner_ips = list(learner_states.keys())
	num_learners = len(learner_ips)
	
	# sorting learner_ips in order of the last digit
	# learner_ip_sort_key = lambda ip : int(ip.split('.')[-1])
	# learner_ips.sort(key=learner_ip_sort_key)

	# shuffling learner_ips
	random.shuffle(learner_ips)

	print('learner_ips:', learner_ips)
	# print('learner_states', learner_states)

	# creates worker handles
	learner_handles = [axon.client.get_RemoteWorker(ip) for ip in learner_ips]
	# print(dir(learner_handles[0].rpcs))

	# This worker composite sends commands to the whole cluster in a single line
	global_cluster = WorkerComposite(learner_handles)
	# benchmark scores is a list of a maps from (task, state) hashes to (training_rate_bps, data_time_spb, param_time_spb) tuples, each index being a learner
	benchmark_scores = await global_cluster.rpcs.get_benchmark_scores()

	# this line creates a new benchmark object that only gives the scores for mnist_cnn for multiple identical tasks
	# benchmark_scores = [multiple_mnist_cnn(learner_scores) for learner_scores in benchmark_scores]

	for learner_scores in benchmark_scores:
		for state in state_names:
			learner_scores[('fashion_2', state)] = learner_scores[('fashion', state)]

	# we want a list of state distributions parallel to the list of benchmark scores
	state_distributions = [learner_states[ip]['state_distribution'] for ip in learner_ips]
	# a list accross learners of lists of learner states, parallel to benchmark scores
	states = [learner_states[ip]['states'] for ip in learner_ips]

	# getting the price per shard for each learner
	worker_prices = [get_random_price() for _ in range(num_learners)]

	print('-----------------------------------------------------------------------')

	# # the variables set by the optimization formulations
	# association, allocation, iterations, time_predictions = None, None, None, None

	# # for task_name in task_names:
	# # 	 tasks[task_name]['deadline'] = tasks[task_name]['deadline'] + float(args.deadline_adjust)
	
	# if (args.data_allocation_regime == 'MED') and (args.state_distribution == 'uncertain'):

	# 	association, allocation, _ = MED(benchmark_scores, worker_prices, state_distributions, tasks)
	# 	# allocation is a float, sometimes with numbers like 5.9999999999999999, which are meant to be 6 but cast down to int 5
	# 	allocation = [round(a) for a in allocation]
	# 	cost = num_GUC*sum([a*p for (a, p) in zip(allocation, worker_prices)])

	# elif (args.data_allocation_regime == 'MMTT') and (args.state_distribution == 'uncertain'):
	# 	# the first states of each learner
	# 	# first_states = [ls[0] for ls in states]
	# 	sampled_states = [random.choices(state_names, weights=state_dist, k=1).pop() for state_dist in state_distributions]
	# 	association, allocation, _ = MMTT_uncertain(benchmark_scores, worker_prices, sampled_states, tasks)
	# 	# allocation is a float, sometimes with numbers like 5.9999999999999999, which are meant to be 6 but cast down to int 5
	# 	allocation = [round(a) for a in allocation]
	# 	cost = num_GUC*sum([a*p for (a, p) in zip(allocation, worker_prices)])

	# elif (args.data_allocation_regime == 'MMTT') and (args.state_distribution == 'ideal'):

	# 	association, allocation = None, None
		
	# 	if args.new_states or not file_exists(learner_state_file):
	# 		# MMTT_ideal is very unstable in that frequently the model is unsolvable, so if we're reinitializing states, we'll try multiple times until we get a combination that is solvable
	# 		num_tries = 0

	# 		while True:
	# 			try:
	# 				print(f'solve attempt {num_tries}')
	# 				association, allocation = MMTT_ideal(benchmark_scores, worker_prices, states, tasks)
	# 				break

	# 			except(AttributeError):
	# 				num_tries += 1
	# 				if num_tries > 10:
	# 					raise BaseException('MMTT_ideal model is unsolvable')

	# 				# resetting learner states
	# 				learner_states = {ip: get_learner_states() for ip in learner_ips}
	# 				states = [learner_states[ip]['states'] for ip in learner_ips]
		
	# 	else:
	# 		# if we're not specifying states this run, we have no choice but to use the saved states and hope they're solvable
	# 		association, allocation = MMTT_ideal(benchmark_scores, worker_prices, states, tasks)

	# 	# allocation is a float, sometimes with numbers like 5.9999999999999999, which are meant to be 6 but cast down to int 5
	# 	allocation = [[round(a) for a in A]	for A in allocation]
	# 	# a is a list of allocations for a single learner, the price of which is just the sum of allocations time the price for that learner
	# 	cost = sum([sum(a)*p for (a, p) in zip(allocation, worker_prices)])

	# else:
	# 	raise(BaseException(f'unrecognized data_allocation_regime {args.data_allocation_regime} {args.state_distribution}'))

	# iterations = [tasks[association[i]]['num_training_iters'] for i in range(num_learners)]
	# results['cost'] = cost

	cost = 10
	association = ['mnist_ffn']
	allocation = [30]
	iterations = [1]
	time_predictions = [30]

	print('-------------------- ')

	print('association:', association)
	print('allocation:', allocation)
	print('cost:', cost)
	# print('time_predictions:', time_predictions)

	# if the optimization calculation fails, there's no need to proceed past this point
	if (association == None):
		exit()

	# we now create composite objects out of the workers assigned to each task, since the tasks can be processed independantly of one another
	# this holds lists of worker handles, split by the task the worker is assigned, all workers in a list work on the same task
	task_piles = {name: [] for name in task_names}
	# this is a parallel dictionary that holds the prediction made for how long each worker will train for
	# prediction_piles = {name: [] for name in task_names}
	# this holds the amount of data allocated to each learner on the task
	data_allocation_piles = {name: [] for name in task_names}

	for i in range(num_learners):
		# the task the learner is assigned
		task_name = association[i]
		# adds that learner to the pile associated with that task
		print(task_name)
		task_piles[task_name].append(learner_handles[i])
		data_allocation_piles[task_name].append(allocation[i])
		# prediction_piles[task_name].append(time_predictions[i])

	# clusters of workers all working on the same task, indexed by the name of the task
	print(task_piles)
	task_clusters = {task_name: WorkerComposite(pile) for task_name, pile in task_piles.items()}

	# results['time_prediction'] = prediction_piles

	training_promises = []
	for task_name in task_clusters:
		training_promises.append(training_routine(task_name, task_clusters[task_name], num_GUC, learner_states, data_allocation_piles[task_name]))

	# deploys training routines in parallel
	await asyncio.gather(*training_promises)

	# records results to a file
	print(result_file_path)
	# record_results(result_file_path)

if (__name__ == '__main__'):
	asyncio.run(main())
