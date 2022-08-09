import axon
import asyncio
import random
import json
import time
import torch
from os.path import join as path_join

from tasks import tasks
from states import state_dicts
from config import config
from data_allocation import EOL, EOL_prime, MMET, RSS
from worker_composite import WorkerComposite
from utils import get_parameter_server

# the number of times each learner will download the model while benchmarking
benchmark_downloads = 5
# the number of shards each learner will train on while benchmarking
benchmark_shards = 10
# the number of shards that is used to assess the testing loss and accuracy of each neural net
testing_shards = 10

task_names = [task_name for task_name in tasks]
state_names = [state_name for state_name in state_dicts]
parameter_server = get_parameter_server()

heat_param = 2

# this is where results will be recorded
result_folder = './results/vary_heat'
result_file = 'TT_2_heat_2.json'
result_file_path = path_join(result_folder, result_file)

# this is the dict where results will be recorded for each training run
# each task_name corresponds to a list of data points, each data point representing a global update cycle
# each data point holds: the time that the cycle took, and the loss and acc
# data_point = {time, acc, loss, sat_ratio, cost}
results = {task_name: [] for task_name in task_names}

def get_state_distribution(heat):
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

async def training_routine(task_name, cluster_handle, num_training_cycles, worker_prices):

	# the number of workers on the given task
	num_learners_on_task = len(cluster_handle.children)

	# assessing the accuracy and loss before training
	loss, acc  = await parameter_server.rpcs.assess_parameters(task_name, testing_shards)
	print(f'the loss and accuracy for {task_name} was: {acc, loss} prior to training')

	data_point = {'time': 0, 'acc': acc, 'loss': loss, 'cost': 0}
	results[task_name].append(data_point)

	for i in range(num_training_cycles):
		# recording the start of the training cycle
		start_time = time.time()

		# randomly sets the state in each of the workers, according to each worker's state_distribution
		# when we want to run tests without state dynamics, pass 'idle' to this function
		# await cluster_handle.rpcs.set_state([('idle', ) for _ in range(num_learners_on_task)])
		await cluster_handle.rpcs.set_state()

		# performs the local training routine on each worker on the task
		await cluster_handle.rpcs.local_update()

		# gets the data allocated to each worker, obtained directly from the workers
		data_allocation = await cluster_handle.rpcs.get_data_allocated()
		# calculates the cost of allocation
		update_cost = sum([price*d for (price, d) in zip(worker_prices, data_allocation)])
		
		# aggregating parameters
		await parameter_server.rpcs.aggregate_parameters(task_name)

		# recording the finish time of the training cycle
		end_time = time.time()

		# assessing parameters
		loss, acc = await parameter_server.rpcs.assess_parameters(task_name, testing_shards)
		print(f'cycle completed on {task_name}, loss and accuracy: {loss, acc}')

		# recording results
		data_point = {'time': end_time - start_time, 'acc': acc, 'loss': loss, 'cost': update_cost}
		results[task_name].append(data_point)

async def main():
	# resets the parameter server from the last training run
	clear_promises = []
	for task_name in task_names:
		clear_promises.append(parameter_server.rpcs.clear_params(task_name))

	await asyncio.gather(*clear_promises)

	learner_ips = axon.discovery.get_ips(ip=config.notice_board_ip)
	num_learners = len(learner_ips)

	print('learner_ips:', learner_ips)

	# creates worker handles
	learner_handles = []
	for ip in learner_ips:
		learner_handles.append(axon.client.RemoteWorker(ip))

	# This worker composite sends commands to the whole cluster
	global_cluster = WorkerComposite(learner_handles)

	# resetting the state distributions of each worker, needed for heat trials
	new_distributions = [(get_state_distribution(heat_param), ) for _ in learner_ips]
	await global_cluster.rpcs.set_state_distribution(new_distributions)

	# benchmark scores is a list of a maps from (task, state) hashes to (training_rate_bps, data_time_spb, param_time_spb) tuples, each index being a learner
	benchmark_scores = await global_cluster.rpcs.get_benchmark_scores()

	# a list of state distributions of each worker
	state_distributions = await global_cluster.rpcs.get_state_distribution()

	# randomly setting the worker prices
	get_random_price = lambda : min(max(random.gauss(config.worker_price_mean, config.worker_price_variance), config.worker_price_min), config.worker_price_max)
	worker_prices = [get_random_price() for _ in range(num_learners)]
	# print(worker_prices)

	# print('benchmark_scores:', benchmark_scores)

	# now allocating data based on benchmark scores

	# get_idle_dist = lambda : [sum([i == 0]) for i in range(len(state_names))]
	# state_distributions = [get_idle_dist() for i in range(num_learners)]
	# association, allocation, iterations = EOL_prime(benchmark_scores, worker_prices, state_distributions)

	association, allocation, iterations = RSS(benchmark_scores, worker_prices)

	print(association)
	print(allocation)
	print(iterations)

	# print(new_distributions)
	# print(state_distributions)

	# exit()

	task_settings = list(zip(association, allocation, iterations))
	await global_cluster.rpcs.set_training_regime(task_settings)

	# we now create composite objects out of the workers assigned to each task, since the tasks can be processed independantly of one another
	# this holds lists of worker handles, split by the task the worker is assigned, all workers in a list work on the same task
	task_piles = {name: [] for name in task_names}
	# same thing for worker prices
	price_piles = {name: [] for name in task_names}

	for i in range(num_learners):
		# the task the learner is assigned
		task_name = association[i]
		# adds that learner to the pile associated with that task
		task_piles[task_name].append(learner_handles[i])
		price_piles[task_name].append(worker_prices[i])

	# clusters of workers all working on the same task, indexed by the name of the task
	task_clusters = {task_name: WorkerComposite(pile) for task_name, pile in task_piles.items()}

	print('budgets', [tasks[task_name]['budget'] for task_name in tasks])

	training_promises = []
	for task_name in task_clusters:
		training_promises.append(training_routine(task_name, task_clusters[task_name],  5, price_piles[task_name]))

	# deploys training routines in parallel
	await asyncio.gather(*training_promises)

	# records results to a file
	record_results(result_file_path)

if (__name__ == '__main__'):
	asyncio.run(main())
