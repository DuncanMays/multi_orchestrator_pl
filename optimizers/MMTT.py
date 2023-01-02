import gurobipy as gurobi
from itertools import product

from tasks import tasks, global_budget
from optimizers.list_utils import get_multilist
from copy import copy

task_names = [name for name in tasks]

# this function manually instantiates an empty 2D list, this can't be done with [None]*5 because of aliasing issues
def get_2D_list(W, H, d=None):
	l = []
	for w in range(W):
		r = []
		for h in range(H):
			r.append(copy(d))
		l.append(r)
	return l

def run_model(workers, requesters, fixed_association=None):
	# ? some sort of configuration object?
	GRB = gurobi.GRB()
	# the model
	m = gurobi.Model('MMTT multi-requester setup')
	m.setParam(GRB.Param.OutputFlag, 0)

	num_workers = len(workers)
	num_requesters = len(requesters)

	# the minimum number of shards assigned to each worker
	delta = 1

	# the 2D list holding the binary decision variables representing if worker j has been assigned tasks from requester i
	x = get_2D_list(num_requesters, num_workers)
	# the 2D list holding the integers representing the amount of data assigned from requester i to worker j
	d = get_2D_list(num_requesters, num_workers)

	# the enumeration of indices of workers and requesters
	combinations = list(product(range(num_requesters), range(num_workers)))
	rw_combinations = list(product(range(num_requesters), range(num_workers)))

	for (i, j) in combinations:
		# worker/requester associations
		x[i][j] = m.addVar(vtype=GRB.BINARY, name="x"+str((i, j)))
		# data allocation
		d[i][j] = m.addVar(vtype=GRB.INTEGER, name="d"+str((i, j)))

	c = [m.addVar(vtype=GRB.INTEGER, name="c"+str(i)) for i in range(num_requesters)]

	# gurobi models update lazily, this executes the addVar statements above
	m.update()

	# this function represents the time that worker l will take on the task t
	delay = lambda t, l : x[t][l]*(2*workers[l].param_times[t] + d[t][l]*workers[l].data_times[t] + requesters[t].num_iters*d[t][l]/workers[l].training_rates[t])

	delay_objective = gurobi.quicksum([ c[i] for i in range(num_requesters) ])

	m.setObjective(delay_objective, GRB.MINIMIZE)

	# we now define contraints

	# if an association is set, fix x to reflect it
	if (fixed_association != None):
		m.addConstrs(( x[i][j] == fixed_association[i][j] for (i, j) in combinations ), 'fix_x')

	# the first constraints i0 to i2, are implicit in the system model

	# that data allocation must be positive
	m.addConstrs(( d[i][j] >= 0 for (i, j) in combinations ), 'i0')
	# x being zero implies d is also zero
	m.addConstrs(( (x[i][j] == 0) >> (d[i][j] == 0) for (i, j) in combinations ), 'i1')
	# x being one means d is greater than zero
	m.addConstrs(( (x[i][j] == 1) >> (d[i][j] >= 1) for (i, j) in combinations ), 'i2')

	# the last constraints are explicitly stated in the problem formulation

	# c1 means the time delay must not exceed the deadline of requester i
	m.addConstrs((delay(i, j) <= c[i] for (i, j) in combinations) , 'c1.0')
	m.addConstrs((c[i] <= requesters[i].deadline for i in range(num_requesters)) , 'c1.1')

	# c2 is an energy constraint

	# c3 means that the total number of data shards assigned from a requester must equal some integer
	m.addConstrs(( sum([ d[i][j] for j in range(num_workers) ]) == requesters[i].dataset_size for i in range(num_requesters)), 'c3')

	# c4 means that the sum of the cost of assignment for all requester may not exceed the budget
	m.addConstr(gurobi.quicksum([ workers[j].price*d[i][j] for i, j in combinations ]) <= global_budget, 'c4')

	# # c5 means that the a worker may only recieve data shards from one requester
	m.addConstrs(( gurobi.quicksum([ x[i][j] for i in range(num_requesters) ] ) <= 1 for j in range(num_workers)), 'c5')

	# # c6 means that the amount of data shards assigned to a worker must be greater than delta
	m.addConstrs((gurobi.quicksum([ d[i][j] for i in range(num_requesters) ]) >= delta for j in range(num_workers)), 'c6')

	# constrains 7 and 8, that the data assignment must be integral, and x must be binary, is implicit in the variable declarations above

	# solves the model
	m.optimize()

	association_2D = get_2D_list(num_requesters, num_workers)
	allocation_2D = get_2D_list(num_requesters, num_workers)

	for (i, j) in combinations:

		# prints an output
		# if (x[i][j].X == 1):
		# 	print(f'worker: {j} got: {d[i][j].X} data slices from requester {i}')

		association_2D[i][j] = x[i][j].X
		allocation_2D[i][j] = d[i][j].X

	# indexes over workers and gives the index of the task they're assigned
	task_indices = [0]*num_workers

	for i in range(num_requesters):
		for j in range(num_workers):
			if (association_2D[i][j] == 1.0):
				task_indices[j] = i

	association = [task_names[i] for i in task_indices]

	# allocation_2D goes over requesters then workers, we want it to go over workers, so that we can lookup from task_indices in each row to get the number of samples for that worker
	allocation_2D_T = [[allocation_2D[i][j] for i in range(num_requesters)] for j in range(num_workers)]
	allocation = [allocation_2D_T[i][task_indices[i]] for i in range(num_workers)]

	ed = get_multilist([num_workers, num_requesters])
	for (r, w) in rw_combinations:
		ed[w][r] = delay(r, w).getValue()

	time_predictions = [ed[i][task_indices[i]] for i in range(num_workers)]

	return association, allocation, time_predictions