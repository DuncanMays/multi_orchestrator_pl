from itertools import product

from data_allocation import EOL, MMET, RSS, EEMO
from tasks import tasks
from states import state_dicts

task_names = [task_name for task_name in tasks]
state_names = [state_name for state_name in state_dicts]
combinations = list(product(task_names, state_names))

num_learners = 4

# benchmark scores is a list of a maps from (task, state) hashes to (training_rate_bps, data_time_spb, param_time_spb) tuples, each index being a learner
benchmark_scores = []

# each learner will be exactly the same, so will have the same benchmark scores
def get_learner_dict():
	learner = {}
	for task, state in combinations:
		learner[(task, state)] = (46.19119831811909, 0.02295656204223633, 0.03764419555664063)

	return learner

if (__name__ == '__main__'):

	for i in range(num_learners):
		benchmark_scores.append(get_learner_dict())

	association, allocation, iterations = EOL(benchmark_scores)
	# association, allocation, iterations = MMET(benchmark_scores)
	# association, allocation, iterations = RSS(benchmark_scores)
	# association, allocation, iterations = EEMO(benchmark_scores)

	print(association)
	print(allocation)
	print(iterations)
