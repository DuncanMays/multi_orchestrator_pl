from itertools import product
import random

from data_allocation import EOL, EOL_prime, MMET, RSS
from tasks import tasks
from states import state_dicts
from config import config

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
		# learner[(task, state)] = (46.19119831811909, 0.02295656204223633, 0.03764419555664063)
		learner[(task, state)] = (5.878207711049711, 0.06955052614212036, 0.12813954353332518)

	return learner

if (__name__ == '__main__'):

	for i in range(num_learners):
		benchmark_scores.append(get_learner_dict())

	from states import state_distribution
	state_probabilities = [state_distribution for _ in range(num_learners)]

	worker_prices = [max(random.gauss(config.worker_price_variance, config.worker_price_mean), 0) for _ in range(num_learners)]

	association, allocation, iterations = RSS(benchmark_scores, worker_prices)
	# association, allocation, iterations = EOL_prime(benchmark_scores, worker_prices, state_probabilities)

	print(association)
	print(allocation)
	print(iterations)

	total_cost = sum([allocation[i]*worker_prices[i] for i in range(num_learners)])

	print(f'total cost of allocation: {total_cost}')
