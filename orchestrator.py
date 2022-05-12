import axon
from config import config
import asyncio

from tasks import tasks
from optimization_formulation import data_allocation
from env import get_worker_obj, get_requester_obj

# the number of times each learner will download the model while benchmarking
benchmark_downloads = 5
# the number of shards each learner will train on while benchmarking
benchmark_shards = 100

task_names = [task_name for task_name in tasks]

async def main():

	learner_ips = axon.discovery.get_ips(ip=config.notice_board_ip)
	num_learners = len(learner_ips)

	print('learner_ips:', learner_ips)

	# creates worker handles
	learner_handles = []
	for ip in learner_ips:
		learner_handles.append(axon.client.RemoteWorker(ip))

	print('running benchmarks')
	benchmark_promises = []
	for task_name in task_names:
		for l in learner_handles:
			benchmark_promises.append(l.rpcs.benchmark(task_name, benchmark_downloads, benchmark_shards))

	# a list of tuples representing the benchmark scores of each worker
	benchmark_scores = await asyncio.gather(*benchmark_promises)

	print('benchmark_scores:', benchmark_scores)

	# now allocating data based on benchmark scores
	# the gurobi script takes input as lists of worker and requester objects

	learner_objs = []
	for i in range(num_learners):

		compute_benchmark = benchmark_scores[i][0]
		comms_benchmark = benchmark_scores[i][1]

		# price, compute benchmark, comms benchmark, k
		learner_obj = get_worker_obj(0.1, compute_benchmark, comms_benchmark, 1)

		learner_objs.append(learner_obj)

	task_objs = []
	for task_name in tasks:

		task = tasks[task_name]

		a = task['num_training_iters']
		b = task['deadline']
		c = task['dataset_size']
		d = task['budget']

		# number of learning iterations, training deadline, data floor, budget
		task_obj = get_requester_obj(a, b, c, d)

		task_objs.append(task_obj)

	print('performing optimization calculation')
	data_allocation(learner_objs, task_objs)

if (__name__ == '__main__'):
	asyncio.run(main())
