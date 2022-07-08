import axon
import asyncio
import random

from tasks import tasks
from states import state_dicts
from config import config
from data_allocation import EOL, MMET, RSS, EEMO
from worker_composite import WorkerComposite

# the number of times each learner will download the model while benchmarking
benchmark_downloads = 5
# the number of shards each learner will train on while benchmarking
benchmark_shards = 10
# the number of shards that is used to assess the testing loss and accuracy of each neural net
testing_shards = 10

task_names = [task_name for task_name in tasks]
state_names = [state_name for state_name in state_dicts]
parameter_server = axon.client.RemoteWorker(config.parameter_server_ip)

async def training_routine(task_name, cluster_handle, num_training_cycles):

	# the number of workers on the given task
	num_learners_on_task = len(cluster_handle.children)

	# assessing the accuracy and loss before training
	acc_and_loss = await parameter_server.rpcs.assess_parameters(task_name, testing_shards)
	print(f'the loss and accuracy for {task_name} was: {acc_and_loss} prior to training')

	for i in range(num_training_cycles):
		# randomly sets the state in each of the workers
		new_states = [(random.choice(state_names), ) for _ in range(num_learners_on_task)]
		await cluster_handle.rpcs.set_state(new_states)

		# performs the local training routine on each worker on the task
		await cluster_handle.rpcs.local_update()
		
		# aggregating parameters
		await parameter_server.rpcs.aggregate_parameters(task_name)

		# assessing parameters
		acc_and_loss = await parameter_server.rpcs.assess_parameters(task_name, testing_shards)
		print(f'cycle completed on {task_name}, loss and accuracy: {acc_and_loss}')

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

	# benchmark scores is a list of a maps from (task, state) hashes to (training_rate_bps, data_time_spb, param_time_spb) tuples, each index being a learner
	benchmark_scores = await global_cluster.rpcs.get_benchmark_scores()

	# print('benchmark_scores:', benchmark_scores)

	# now allocating data based on benchmark scores
	association, allocation, iterations = EOL(benchmark_scores)
	# association, allocation, iterations = MMET(benchmark_scores)
	# association, allocation, iterations = RSS(benchmark_scores)
	# association, allocation, iterations = EEMO(benchmark_scores)

	print(association)
	print(allocation)
	print(iterations)

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
		training_promises.append(training_routine(task_name, task_clusters[task_name], 10))

	# deploys training routines in parallel
	await asyncio.gather(*training_promises)

if (__name__ == '__main__'):
	asyncio.run(main())
