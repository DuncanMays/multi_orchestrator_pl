# most of the hard-number crunching is done in the optimization framework gurobi
# gurobi optimization formulations are contained in other files and imported
# this file essentially functions as a translation layer between benchmarking scores and the format gurobi understands

import torch
import random
from types import SimpleNamespace
from tasks import tasks
from states import state_dicts

from EOL_formulation import run_model as EOL_model
from RSS_formulation import run_model as RSS_model

task_names = [task_name for task_name in tasks]
state_names = [state_name for state_name in state_dicts]

def EOL(benchmark_scores):

	learner_objs = []
	for learner_scores in benchmark_scores:
		# learner_scores is a map from from (task, state) names to (training_rate_bps, data_time_spb, param_time_spb) tuples

		compute_benchmarks = {task_state_hash: learner_scores[task_state_hash][0] for task_state_hash in learner_scores}
		data_times = {task_state_hash: learner_scores[task_state_hash][1] for task_state_hash in learner_scores}
		param_times = {task_state_hash: learner_scores[task_state_hash][2] for task_state_hash in learner_scores}

		learner_obj = SimpleNamespace(**{
			'price': 0.1,
			'kappa': 1,
			'training_rates': compute_benchmarks,
			'data_times': data_times,
			'param_times': param_times
		})

		learner_objs.append(learner_obj)

	task_objs = []
	for task_name in tasks:

		task = tasks[task_name]

		# number of learning iterations, training deadline, data floor, budget
		task_obj = SimpleNamespace(**{
			'num_iters': task['num_training_iters'] ,
			'T': task['deadline'],
			'dataset_size': task['dataset_size'],
			'budget': task['budget']
		})

		task_objs.append(task_obj)

	print('performing optimization calculation')
	# returns the learner/orchestrator association as a one-hot matrix, and the data allocated to each learner as a matrix as well
	# the first index iterates accross requesters, the second accross workers
	x, d = EOL_model(learner_objs, task_objs)

	# a tensor that holds true on index pairs of workers and requesters who are associated with one another
	x = torch.tensor(x) == 1.0

	task_indices = x.nonzero()[:, 0].tolist()

	association = [task_names[i] for i in task_indices]

	# the amount of data allocated to each learner
	allocation = torch.tensor(d).sum(dim=0).tolist()

	return association, allocation

def RSS(benchmark_scores):

	learner_objs = []
	for learner_scores in benchmark_scores:
		# learner_scores is a map from from (task, state) names to (training_rate_bps, data_time_spb, param_time_spb) tuples

		# samples benchmarks from a random state
		state = random.choice(state_names)

		compute_benchmarks = {task_names.index(task) : learner_scores[(task, state)][0] for task in task_names}
		data_times = {task_names.index(task) : learner_scores[(task, state)][1] for task in task_names}
		param_times = {task_names.index(task) : learner_scores[(task, state)][2] for task in task_names}

		learner_obj = SimpleNamespace(**{
			'price': 0.1,
			'kappa': 1,
			'training_rates': compute_benchmarks,
			'data_times': data_times,
			'param_times': param_times
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
	x, d = RSS_model(learner_objs, task_objs)

	# a tensor that holds true on index pairs of workers and requesters who are associated with one another
	x = torch.tensor(x) == 1.0

	task_indices = x.nonzero()[:, 0].tolist()

	association = [task_names[i] for i in task_indices]

	# the amount of data allocated to each learner
	allocation = torch.tensor(d).sum(dim=0).tolist()

	return association, allocation