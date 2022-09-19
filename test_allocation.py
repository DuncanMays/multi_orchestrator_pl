from itertools import product
import random
import torch

from data_allocation import EOL_prime, RSS, EOL_minmax, MED, MMTT
from tasks import tasks
from states import state_dicts
from config import config

task_names = [task_name for task_name in tasks]
state_names = [state_name for state_name in state_dicts]
combinations = list(product(task_names, state_names))

num_learners = 9

# benchmark scores is a list of a maps from (task, state) hashes to (training_rate_bps, data_time_spb, param_time_spb) tuples, each index being a learner
benchmark_scores = []

def get_learner_scores():
	return (
		5.878207711049711 + random.gauss(0, 1),
		max(0.06955052614212036 + random.gauss(0, 0.1), 0.01),
		max(0.12813954353332518 + random.gauss(0, 0.2), 0.01)
	)

# each learner will be exactly the same, so will have the same benchmark scores
def get_learner_dict():
	learner = {}
	for task, state in combinations:
		learner[(task, state)] = get_learner_scores()

	return learner

get_idle_dist = lambda : [sum([i == 0]) for i in range(len(state_names))]

def get_state_distribution():
	# state_heat controls how spread out the worker's state distribution is, hotter distributions are more spread out, colder ones more concentrated on one state
	state_heat = 1/2
	# this is the distribution that controls how workers change states
	state_distribution = torch.softmax(torch.randint(0, 2, (len(state_names), ))/state_heat, dim=0).tolist()

	return state_distribution

if (__name__ == '__main__'):

	for i in range(num_learners):
		benchmark_scores.append(get_learner_dict())

	state_probabilities = [get_idle_dist() for _ in range(num_learners)]

	worker_prices = [max(random.gauss(config.worker_price_variance, config.worker_price_mean), 0) for _ in range(num_learners)]

	# association, allocation, iterations, EOL = RSS(benchmark_scores, worker_prices, state_probabilities)
	# association, allocation, iterations, EOL = MMTT(benchmark_scores, worker_prices, state_probabilities)
	# association, allocation, iterations, EOL = EOL_prime(benchmark_scores, worker_prices, state_probabilities)
	# association, allocation, iterations, EOL = EOL_minmax(benchmark_scores, worker_prices, state_probabilities)
	association, allocation, iterations, EOL = MED(benchmark_scores, worker_prices, state_probabilities)

	print('--------------------------------------------------------------------------')
	print(association)
	print(allocation)
	print(iterations)
	print(EOL)

	total_cost = sum([allocation[i]*worker_prices[i] for i in range(num_learners)])

	print(f'total cost of allocation: {total_cost}')
