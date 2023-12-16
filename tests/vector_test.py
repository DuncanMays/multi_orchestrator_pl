# this file exists to test the statistics of our allocation methods MED and MMTT at a very basic level

# from matplotlib import pyplot as plt
import random

num_learners = 5
num_samples = 100

def max_product(x, y):
	return max([a*b for a, b in zip(x, y)])

def get_random_x():
	return [random.uniform(1, 10) for _ in range(num_learners)]

def MED():
	# returns a uniform allocation
	return [num_samples/num_learners for _ in range(num_learners)]

def MMTT():
	# randomly samples x
	x_prime = get_random_x()
	reciprocal_sum = sum([1/a for a in x_prime])
	allocation = [num_samples/(reciprocal_sum*a) for a in x_prime]
	return allocation

def main():
	med_wins = []
	num_trials = 500

	for i in range(num_trials):
		x = get_random_x()
		med_allocation = MED()
		mmtt_allocation = MMTT()

		# print(max_product(x, mmtt_allocation), max_product(x, med_allocation))
		# print(max_product(x, mmtt_allocation) > max_product(x, med_allocation))
		med_wins.append(max_product(x, mmtt_allocation)/max_product(x, med_allocation))

	print(sum(med_wins)/num_trials)

if __name__ == '__main__':
	main()