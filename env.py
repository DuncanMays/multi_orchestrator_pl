from types import SimpleNamespace
import random

# inputs are: the worker's price, compute benchmark score, communication benchmark score, and kappa value
def get_worker_obj(p=None, b=None, c=None, k=None):

	# overwrites parameters with default values if not given
	if (p == None):
		p = random.choice([i/10 for i in range(1, 5)])
	if (b == None):
		b = random.randint(100, 1000)
	if (c == None):
		c = random.randint(100, 1000)
	if (k == None):
		k = 1

	w = {
		# the price
		'p': p,
		# compute benchmark
		'b': b,
		# communication benchmark
		'c': c,
		# something with power consumption
		'k': k
	}

	return SimpleNamespace(**w)

# arguements are: number of learning iterations, training deadline, data floor, budget
def get_requester_obj(mu=None, T=None, D=None, B=None):

	if (mu == None):
		mu = random.choice([1, 1, 2])
	if (T == None):
		T = random.randint(45, 75)
	if (D == None):
		D = random.randint(100, 1000)
	if (B == None):
		B = random.randint(50_000, 50_001)


	r = {
		# the number of learning iterations
		'mu': mu,
		# the training deadline
		'T': T,
		# the data floor
		'D': D,
		# the requester's budget
		'B': B
	}

	return SimpleNamespace(**r)