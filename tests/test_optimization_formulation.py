from types import SimpleNamespace
import sys

sys.path.append('..')

from optimization_formulation import data_allocation
from tasks import tasks
import random

def test_data_allocation(num_learners, tasks):

	learner_objs = []
	for i in range(num_learners):

		compute_benchmark = random.randint(0, 100)
		data_time = random.randint(0, 10)
		param_time = random.randint(0, 10)

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

	data_allocation(learner_objs, task_objs)

def main():
	test_data_allocation(10, tasks)

if (__name__ == '__main__'):
	main()