import gurobi
from itertools import product

from states import state_dicts
from tasks import tasks, global_budget
from optimizers.list_utils import get_2D_list, get_multilist

task_names = [name for name in tasks]
state_names = [name for name in state_dicts]

def run_model(workers, requesters, state_probabilities):
	# ? some sort of configuration object?
	GRB = gurobi.GRB()
	# the model
	m = gurobi.Model('multi-requester setup')
	m_prime = gurobi.Model('multi-requester setup helper')

	num_requesters = len(requesters)
	num_workers = len(workers)
	num_states = len(state_dicts)

	# the minimum number of shards assigned to each worker
	delta = 1

	# the 2D list holding the binary decision variables representing if worker j has been assigned tasks from requester i
	x = get_2D_list(num_requesters, num_workers)
	# the 2D list holding the integers representing the amount of data assigned from requester i to worker j
	d = get_2D_list(num_requesters, num_workers)
	# the 3D list used to calculate the optimal allocation for EOL
	d_prime = get_multilist([num_requesters, num_workers, num_states])
	x_prime = get_multilist([num_requesters, num_workers, num_states])

	# the enumeration of indices of workers, requesters, and states
	rws_combinations = list(product(range(num_requesters), range(num_workers), range(num_states)))
	# likewise for workers/requesters, requesters/states, and workers/states
	rw_combinations = list(product(range(num_requesters), range(num_workers)))
	rs_combinations = list(product(range(num_requesters), range(num_states)))
	ws_combinations = list(product(range(num_workers), range(num_states)))

	for (i, j) in rw_combinations:
		# worker/requester associations
		x[i][j] = m.addVar(vtype=GRB.BINARY, name="x"+str((i, j)))
		# data allocation
		d[i][j] = m.addVar(vtype=GRB.INTEGER, name="d"+str((i, j)))

	for (r, w, s) in rws_combinations:
		x_prime[r][w][s] = m_prime.addVar(vtype=GRB.BINARY, name="x_prime"+str((r, w, s)))
		d_prime[r][w][s] = m_prime.addVar(vtype=GRB.INTEGER, name="d_prime"+str((r, w, s)))

	# gurobi models update lazily, this executes the addVar statements above
	m.update()
	m_prime.update()

	# this function represents the time that worker l will take on the task t with tsh being the task/state hash
	delay = lambda t, l, tsh : x[t][l]*(2*workers[l].param_times[tsh] + d[t][l]*workers[l].data_times[tsh] + requesters[t].num_iters*d[t][l]/workers[l].training_rates[tsh])

	# this function takes task, worker and state index and returns the delay according to the prime data distribution
	def delay_prime(t, w, s):
		tsh = (task_names[t], state_names[s])
		return x_prime[t][w][s]*(2*workers[w].param_times[tsh] + d_prime[t][w][s]*workers[w].data_times[tsh] + requesters[t].num_iters*d_prime[t][w][s]/workers[w].training_rates[tsh])

	# this function represents the expected time that worker j will take to evaluate the learning task assigned to them from requester i
	def expected_delay(i, j):
		# expected delay
		ed = 0
		state_names = list(state_dicts.keys())

		# iterates over states, summing their delay multiplied by their probability
		for state_index in range(num_states):
			state_name = state_names[state_index]
			tsh = (task_names[i], state_name)
			ed += state_probabilities[j][state_index]*delay(i, j, tsh)

		return ed


	# constraints for the prime formulation:
	
	# that data allocation must be positive
	m_prime.addConstrs(( d_prime[i][j][k] >= 0 for (i, j, k) in rws_combinations ), 'i0_prime')
	# x being zero implies d is also zero
	m_prime.addConstrs(( (x_prime[r][w][s] == 0) >> (d_prime[r][w][s] == 0) for (r, w, s) in rws_combinations ), 'i1_prime')
	# x being one means d is greater than zero
	m_prime.addConstrs(( (x_prime[r][w][s] == 1) >> (d_prime[r][w][s] >= 1) for (r, w, s) in rws_combinations ), 'i2_prime')
	# c1 means the time delay must not exceed the deadline of requester
	m_prime.addConstrs((delay_prime(r, w, s) <= requesters[r].T for (r, w, s) in rws_combinations) , 'c1_prime')
	# means that the total number of data shards assigned from a requester must equal some integer
	m_prime.addConstrs(( sum([ d_prime[r][w][s] for w in range(num_workers) ]) == requesters[r].dataset_size for (r, s) in rs_combinations), 'c3_prime')
	# means that the total cost of assignment for a requester in each state may not exceed their budget
	m_prime.addConstrs((sum([ workers[w].price*d_prime[r][w][s] for (r, w) in rw_combinations ]) <= global_budget for s in range(num_states)), 'c4')
	# means that the amount of data shards assigned to a worker must be greater than delta
	m_prime.addConstrs((gurobi.quicksum([ d_prime[r][w][s] for r in range(num_requesters) ]) >= delta for (w, s) in ws_combinations), 'c6')


	prime_objective = gurobi.quicksum([ gurobi.quicksum([
			state_probabilities[w][s]*delay_prime(r, w, s) for (w, s) in ws_combinations
		]) for r in range(num_requesters) 
	])

	m_prime.setObjective(prime_objective, GRB.MINIMIZE)

	m_prime.optimize()

	# we now define contraints for the main formulation

	# the first constraints i0 to i2, are implicit in the system model

	# that data allocation must be positive
	m.addConstrs(( d[i][j] >= 0 for (i, j) in rw_combinations ), 'i0')
	# x being zero implies d is also zero
	m.addConstrs(( (x[i][j] == 0) >> (d[i][j] == 0) for (i, j) in rw_combinations ), 'i1')
	# x being one means d is greater than zero
	m.addConstrs(( (x[i][j] == 1) >> (d[i][j] >= 1) for (i, j) in rw_combinations ), 'i2')

	# the last constraints are explicitly stated in the problem formulation

	# c1 means the time delay must not exceed the deadline of requester i
	m.addConstrs((expected_delay(i, j) <= requesters[i].T for (i, j) in rw_combinations) , 'c1')
	# c2 is an energy constraint
	# c3 means that the total number of data shards assigned from a requester must equal some integer
	m.addConstrs(( sum([ d[r][w] for w in range(num_workers) ]) == requesters[r].dataset_size for r in range(num_requesters)), 'c3')
	# c4 means that the total cost of assignment for a requester may not exceed their budget
	# m.addConstrs((sum([ workers[j].price*d[i][j] for j in range(num_workers) ]) <= requesters[i].budget for i in range(num_requesters)), 'c4')
	m.addConstr(gurobi.quicksum([ workers[j].price*d[i][j] for i, j in rw_combinations ]) <= global_budget, 'c4')
	# # c5 means that the a worker may only recieve data shards from one requester
	m.addConstrs(( gurobi.quicksum([ x[i][j] for i in range(num_requesters) ] ) <= 1 for j in range(num_workers)), 'c5')
	# # c6 means that the amount of data shards assigned to a worker must be greater than delta
	m.addConstrs((gurobi.quicksum([ d[i][j] for i in range(num_requesters) ]) >= delta for j in range(num_workers)), 'c6')

	# constrains 7 and 8, that the data assignment must be integral, and x must be binary, is implicit in the variable declarations above

	# delay_prime(r, w, s).getValue() represents the right hand term, the minimum delay of allocation across all workers in the given state on the given task

	EOL_objective = gurobi.quicksum([ gurobi.quicksum([
		expected_delay(r, w) for w in range(num_workers)]) - gurobi.quicksum([
			state_probabilities[w][s]*delay_prime(r, w, s).getValue() for (w, s) in ws_combinations
		]) for r in range(num_requesters) 
	])

	m.setObjective(EOL_objective, GRB.MINIMIZE)

	m.optimize()

	association = get_2D_list(num_requesters, num_workers)
	allocation = get_2D_list(num_requesters, num_workers)

	for (i, j) in rw_combinations:

		# prints an output
		# if (x[i][j].X == 1):
		# 	print(f'worker: {j} got: {d[i][j].X} data slices from requester {i}')

		association[i][j] = x[i][j].X
		allocation[i][j] = d[i][j].X

	return association, allocation, EOL_objective.getValue()