import gurobi
from itertools import product
from functools import lru_cache

from states import state_dicts
from tasks import tasks, global_budget
from optimizers.list_utils import get_2D_list, get_multilist

task_names = [name for name in tasks]
state_names = [name for name in state_dicts]

def run_model(workers, requesters, state_probabilities):
	print('MED')
	# ? some sort of configuration object?
	GRB = gurobi.GRB()
	# the model
	m = gurobi.Model('MED multi-requester setup')
	m.setParam(GRB.Param.OutputFlag, 0)

	# the minimum number of shards assigned to each worker
	delta = 1

	num_requesters = len(requesters)
	num_workers = len(workers)
	num_states = len(state_dicts)

	# the list of all possible state tuples for the cluster
	state_combinations = list(product(*(range(num_states) for _ in range(num_workers))))
	num_state_combinations = len(state_combinations)

	# the 2D list holding the binary decision variables representing if worker j has been assigned tasks from requester i
	x = get_2D_list(num_requesters, num_workers)
	# the 2D list holding the integers representing the amount of data assigned from requester i to worker j
	d = get_2D_list(num_requesters, num_workers)

	# ceiling variables
	c = get_multilist([num_requesters, num_state_combinations])

	# the enumeration of indices of workers, requesters, and states
	# rws_combinations = list(product(range(num_requesters), range(num_workers), range(num_state_combinations)))
	# likewise for workers/requesters, requesters/states, and workers/states
	rw_combinations = list(product(range(num_requesters), range(num_workers)))
	# rs_combinations = list(product(range(num_requesters), range(num_state_combinations)))
	# ws_combinations = list(product(range(num_workers), range(num_state_combinations)))

	# print('-+-+-+----------------------+-+-+-', num_state_combinations)

	for (i, j) in rw_combinations:
		# worker/requester associations
		x[i][j] = m.addVar(vtype=GRB.BINARY, name="x"+str((i, j)))
		# data allocation
		d[i][j] = m.addVar(vtype=GRB.INTEGER, name="d"+str((i, j)))

	for r in range(num_requesters):
		c[r] = m.addVar(vtype=GRB.INTEGER, name='c'+str(r))

	# gurobi models update lazily, this executes the addVar statements above
	m.update()

	# this function represents the time that worker l will take on the task t with tsh being the task/state hash
	delay = lambda t, l, tsh : x[t][l]*(2*workers[l].param_times[tsh] + d[t][l]*workers[l].data_times[tsh] + requesters[t].num_iters*d[t][l]/workers[l].training_rates[tsh])

	# the first constraints i0 to i2, are implicit in the system model

	# that data allocation must be positive
	m.addConstrs(( d[i][j] >= 0 for (i, j) in rw_combinations ), 'i0')
	# x being zero implies d is also zero
	m.addConstrs(( (x[i][j] == 0) >> (d[i][j] == 0) for (i, j) in rw_combinations ), 'i1')
	# x being one means d is greater than zero
	m.addConstrs(( (x[i][j] == 1) >> (d[i][j] >= 1) for (i, j) in rw_combinations ), 'i2')

	# the last constraints are explicitly stated in the problem formulation

	# the delay of a given task/worker in a given state
	@lru_cache(maxsize=num_requesters*num_workers*num_states)
	def delay(r, w, s):
		tsh = (task_names[r], state_names[s])
		delay = x[r][w]*(2*workers[w].param_times[tsh] + d[r][w]*workers[w].data_times[tsh] + requesters[r].num_iters*d[r][w]/workers[w].training_rates[tsh])
		return delay

	# this function represents the expected time that worker j will take to evaluate the learning task assigned to them from requester i
	def expected_delay(i, j):
		# expected delay
		ed = 0

		# iterates over states, summing their delay multiplied by their probability
		for state_index in range(num_states):
			ed += state_probabilities[j][state_index]*delay(i, j, state_index)

		return ed

	# c[r][s] should be the maximal delay of all workers association with r given the state tuple s
	# this constraint means that c[i][s_bar] is the maximal delay of all workers associated with task i for every given state tuple s_bar
	m.addConstrs((expected_delay(r, w) <= c[r] for (r, w) in rw_combinations) , 'c1.0')
	# c1 means the expected time delay must not exceed the deadline of requester i
	m.addConstrs(( c[r] <= requesters[r].T for r in range(num_requesters) ), 'c1.1')
	# m.addConstrs(( expected_delay(r, w) <= requesters[r].T for (r, w) in rw_combinations ), 'c1.1')

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

	MED_objective = gurobi.quicksum([ c[r] for r in range(num_requesters) ])

	m.setObjective(MED_objective, GRB.MINIMIZE)

	m.optimize()

	# the delay of a given task/worker in a given state, as calculated from the solved model
	@lru_cache(maxsize=num_requesters*num_workers*num_states)
	def delay_real(r, w, s):
		tsh = (task_names[r], state_names[s])
		delay = x[r][w].X*(2*workers[w].param_times[tsh] + d[r][w].X*workers[w].data_times[tsh] + requesters[r].num_iters*d[r][w].X/workers[w].training_rates[tsh])
		return delay

	# this function represents the expected time that worker j will take to evaluate the learning task assigned to them from requester i, as calculated from the solved model
	def expected_delay_real(i, j):
		# expected delay
		ed = 0

		# iterates over states, summing their delay multiplied by their probability
		for state_index in range(num_states):
			# print('-----------------------------------')
			# print(delay_real(i, j, state_index))
			# print(state_probabilities[j][state_index])
			ed += state_probabilities[j][state_index]*delay_real(i, j, state_index)

		print('-----------------------------------')
		print(ed)
		return ed

	association_2D = get_2D_list(num_requesters, num_workers)
	allocation_2D = get_2D_list(num_requesters, num_workers)

	for (i, j) in rw_combinations:

		# prints an output
		# if (x[i][j].X == 1):
		# 	print(f'worker: {j} got: {d[i][j].X} data slices from requester {i}')

		association_2D[i][j] = x[i][j].X
		allocation_2D[i][j] = d[i][j].X

	# indexes over workers and gives the index of the task they're assigned
	task_indices = [0]*num_workers

	for i in range(len(task_names)):
		for j in range(num_workers):
			if (association_2D[i][j] == 1.0):
				task_indices[j] = i

	association = [task_names[i] for i in task_indices]

	# allocation_2D goes over requesters then workers, we want it to go over workers, so that we can lookup from task_indices in each row to get the number of samples for that worker
	allocation_2D_T = [[allocation_2D[i][j] for i in range(num_requesters)] for j in range(num_workers)]
	allocation = [allocation_2D_T[i][task_indices[i]] for i in range(num_workers)]

	ed = get_multilist([num_workers, num_requesters])
	for (r, w) in rw_combinations:
		ed[w][r] = expected_delay(r, w).getValue()

	time_predictions = [ed[i][task_indices[i]] for i in range(num_workers)]

	return association, allocation, time_predictions