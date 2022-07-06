import axon
import asyncio
import random

from tasks import tasks
from states import state_dicts
from config import config
from data_allocation import EOL, RSS
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
	
# this coroutine aggregates parameters for a certain task and then assesses their loss and accruacy
async def aggregate_and_assess(task_name):
	await parameter_server.rpcs.aggregate_parameters(task_name)
	return await parameter_server.rpcs.assess_parameters(task_name, testing_shards)

async def main():
	# resets the parameter server from the last training routine
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

	cluster = WorkerComposite(learner_handles)

	# benchmark scores is a list of a maps from (task, state) hashes to (training_rate_bps, data_time_spb, param_time_spb) tuples, each index being a learner
	benchmark_scores = await cluster.rpcs.get_benchmark_scores()

	# print('benchmark_scores:', benchmark_scores)

	# now allocating data based on benchmark scores
	association, allocation = EOL(benchmark_scores)
	# association, allocation = RSS(benchmark_scores)

	print(allocation)
	print(association)

	data_set_promises = []
	for i in range(num_learners):
		learner = learner_handles[i]

		task_name = association[i]
		num_shards = allocation[i]

		print('task_name, num_shards', task_name, num_shards)
		data_set_promises.append(learner.rpcs.set_training_regime(incoming_task_name=task_name, incomming_num_shards=num_shards))

	# waits for the promises that set the task on each learner to resolve
	await asyncio.gather(*data_set_promises)

	# assessing parameters for each task prior to training
	acc_and_loss_pending = []
	for task_name in task_names:
		acc_and_loss_pending.append(parameter_server.rpcs.assess_parameters(task_name, testing_shards))

	acc_and_loss = await asyncio.gather(*acc_and_loss_pending)

	for i in range(len(task_names)):
		print(f'the loss and accuracy for {task_names[i]} was: {acc_and_loss[i]}')

	# performs local updates
	for i in range(100):

		# randomly sets the state in each of the workers
		new_states = [(random.choice(state_names), ) for _ in range(num_learners)]
		await cluster.rpcs.set_state(new_states)

		await cluster.rpcs.local_update()
		
		# aggregating and then assessing parameters for each task on the parameter server
		acc_and_loss_pending = []
		for task_name in task_names:
			acc_and_loss_pending.append(aggregate_and_assess(task_name))

		acc_and_loss = await asyncio.gather(*acc_and_loss_pending)

		for i in range(len(task_names)):
			print(f'the loss and accuracy for {task_names[i]} was: {acc_and_loss[i]}')

if (__name__ == '__main__'):
	asyncio.run(main())
