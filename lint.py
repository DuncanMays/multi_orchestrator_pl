# this function sets all the values that vary in between experiments, given the same set of workers between experiments
# the two peices of information that change are the worker state distributions and the worker prices, both of which are returned by this function
def initialize_parameters(num_learners):
	worker_prices = [get_random_price() for _ in range(num_learners)]

	state_distributions = None

	if args.state_distribution == 'ideal':
		# state distributions are ideal
		# state_distributions = [get_idle_dist() for i in range(num_learners)]

		# creating state distributions for each learner
		prime_state_dists = [create_state_distribution(float(args.heat)) for _ in range(num_learners)]
		# the state distribution that's sent to learners will be a one-hot vector representing the state sampled from the corresponding distributions initialized above
		# this will mean the worker's state is randomly set but constant, or static
		states = [random.choices(range(len(state_names)), weights=state_dist, k=1).pop() for state_dist in prime_state_dists]
		state_distributions = [get_one_hot(s) for s in states]

	elif args.state_distribution == 'uncertain':
		# randomly setting the state distributions of each worker, based on heat parameter
		state_distributions = [create_state_distribution(args.heat) for _ in range(num_learners)]

	else:
		raise BaseException('unknown state_distribution: ', args.state_distribution)

	return worker_prices, state_distributions