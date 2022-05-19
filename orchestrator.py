import axon
from config import config
import asyncio
from types import SimpleNamespace
import torch

from tasks import tasks
from optimization_formulation import data_allocation

# the number of times each learner will download the model while benchmarking
benchmark_downloads = 5
# the number of shards each learner will train on while benchmarking
benchmark_shards = 10
# the number of shards that is used to assess the testing loss and accuracy of each neural net
testing_shards = 10

task_names = [task_name for task_name in tasks]

parameter_server = axon.client.RemoteWorker(config.parameter_server_ip)

model_map = {}
for task_name in tasks:
	model_map[task_name] = tasks[task_name]['network_architecture']()

async def global_update_cycle(learner_handles):

	# performing local updates
	update_promises = []
	for learner in learner_handles:
		update_promises.append(learner.rpcs.local_update())

	await asyncio.gather(*update_promises)

	# aggregating parameters
	aggregate_promises = []
	for task_name in task_names:
		aggregate_promises.append(parameter_server.rpcs.aggregate_parameters(task_name))

	await asyncio.gather(*aggregate_promises)

async def assess_parameters():
	testing_promises = []
	for task_name in task_names:
		testing_promises.append(parameter_server.rpcs.assess_parameters(task_name, testing_shards))

	acc_and_loss = await asyncio.gather(*testing_promises)

	for i in range(len(task_names)):
		print(f'the loss and accuracy for {task_names[i]} was: {acc_and_loss[i]}')

async def main():

	learner_ips = axon.discovery.get_ips(ip=config.notice_board_ip)
	num_learners = len(learner_ips)

	print('learner_ips:', learner_ips)

	# creates worker handles
	learner_handles = []
	for ip in learner_ips:
		learner_handles.append(axon.client.RemoteWorker(ip))

	print('starting workers')
	startup_promises = []
	for l in learner_handles:
		startup_promises.append(l.rpcs.startup())

	await asyncio.gather(*startup_promises)

	exit()

	print('benchmark_scores:', benchmark_scores)

	# now allocating data based on benchmark scores
	# the gurobi script takes input as lists of worker and requester objects

	learner_objs = []
	for i in range(num_learners):

		compute_benchmark = benchmark_scores[i][0]
		data_time = benchmark_scores[i][1]
		param_time = benchmark_scores[i][2]

		learner_obj = SimpleNamespace(**{
			'price': 0.1,
			'kappa': 1,
			'training_rate': compute_benchmark,
			'data_time': data_time,
			'param_time': param_time
		})

		learner_objs.append(learner_obj)

	task_objs = []
	for task_name in tasks:

		task = tasks[task_name]

		# number of learning iterations, training deadline, data floor, budget
		task_obj = SimpleNamespace(**{
			'num_iters': task['num_training_iters'] ,
			'deadline': task['deadline'],
			'dataset_size': task['dataset_size'],
			'budget': task['budget']
		})

		task_objs.append(task_obj)

	print('performing optimization calculation')
	# returns the learner/orchestrator association as a one-hot matrix, and the data allocated to each learner as a matrix as well
	# the first index iterates accross requesters, the second accross workers
	x, d = data_allocation(learner_objs, task_objs)

	# a tensor that holds true on index pairs of workers and requesters who are associated with one another
	x = torch.tensor(x) == 1.0

	task_indices = x.nonzero()[:, 0].tolist()

	association = [task_names[i] for i in task_indices]

	# the amount of data allocated to each learner
	allocation = torch.tensor(d).sum(dim=0).tolist()

	print(allocation)
	print(association)

	# resets the parameter server from the last training regime
	clear_promises = []
	for task_name in task_names:
		clear_promises.append(parameter_server.rpcs.clear_params(task_name))

	await asyncio.gather(*clear_promises)

	data_set_promises = []
	for i in range(num_learners):
		learner = learner_handles[i]

		task_name = association[i]
		num_shards = allocation[i]

		print('task_name, num_shards', task_name, num_shards)
		data_set_promises.append(learner.rpcs.set_training_regime(incoming_task_name=task_name, incomming_num_shards=num_shards))

	data_set_promises.append(assess_parameters())

	# waits for the promises that set the task on each learner to resolve
	await asyncio.gather(*data_set_promises)

	for i in range(10):
		await global_update_cycle(learner_handles)
		await assess_parameters()

if (__name__ == '__main__'):
	asyncio.run(main())
