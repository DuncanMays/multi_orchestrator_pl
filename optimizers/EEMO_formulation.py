import gurobi
from itertools import product

from states import state_dicts
from tasks import tasks

task_names = [name for name in tasks]
state_names = [name for name in state_dicts]

# this function manually instantiates an empty 2D list, this can't be done with [None]*5 because of aliasing issues
def get_2D_list(W, H):
	l = []
	for w in range(W):
		r = []
		for h in range(H):
			r.append(None)
		l.append(r)
	return l

def run_model(workers, requesters):
	# ? some sort of configuration object?
	GRB = gurobi.GRB()
	# the model
	m = gurobi.Model('multi-requester setup')

	m.params.NonConvex = 2
	m.setParam('NonConvex', 2)

	num_workers = len(workers)
	num_requesters = len(requesters)
	num_states = len(state_dicts)

	# the minimum number of shards assigned to each worker
	delta = 1

	# the 2D list holding the binary decision variables representing if worker j has been assigned tasks from requester i
	x = get_2D_list(num_requesters, num_workers)
	# the 2D list holding the integers representing the amount of data assigned from requester i to worker j
	d = get_2D_list(num_requesters, num_workers)

	# the enumeration of indices of workers and requesters
	combinations = list(product(range(num_requesters), range(num_workers)))

	for (i, j) in combinations:
		# worker/requester associations
		x[i][j] = m.addVar(vtype=GRB.BINARY, name="x"+str((i, j)))
		# data allocation
		d[i][j] = m.addVar(vtype=GRB.INTEGER, name="d"+str((i, j)))

	# the number of learning iterations each worker is to perform
	num_iters = [m.addVar(vtype=GRB.INTEGER, name="i"+str(i)) for i in range(num_requesters)]

	# gurobi models update lazily, this executes the addVar statements above
	m.update()

	# this function represents the time that worker l will take on the task t
	delay = lambda t, l : 2*workers[l].param_times[t] + d[t][l]*workers[l].data_times[t] + num_iters[t]*d[t][l]/workers[l].training_rates[t]

	# This represents the objective of optimization, to maximize the number of iterations each learner performs in the time available
	learning_objective = gurobi.quicksum([ t for t in num_iters ])

	m.setObjective(learning_objective, GRB.MAXIMIZE)

	# we now define contraints

	# the first constraints i0 to i3, are implicit in the system model

	# that data allocation must be positive
	m.addConstrs(( d[i][j] >= 0 for (i, j) in combinations ), 'i0')
	# x being zero implies d is also zero
	m.addConstrs(( (x[i][j] == 0) >> (d[i][j] == 0) for (i, j) in combinations ), 'i1')
	# x being one means d is greater than zero
	m.addConstrs(( (x[i][j] == 1) >> (d[i][j] >= 1) for (i, j) in combinations ), 'i2')
	# that the number of training iterations must be greater than or equal to one
	m.addConstrs(( num_iters[i] >= 1 for i in range(num_requesters) ), 'i3')

	# the last constraints are explicitly stated in the problem formulation

	# c1 means the time delay must not exceed the deadline of requester i
	m.addConstrs((delay(i, j) <= requesters[i].deadline for (i, j) in combinations) , 'c1')

	# c2 is an energy constraint

	# c3 means that the total number of data shards assigned from a requester must equal some integer
	m.addConstrs(( sum([ d[i][j] for j in range(num_workers) ]) == requesters[i].dataset_size for i in range(num_requesters)), 'c3')

	# c4 means that the total cost of assignment for a requester may not exceed their budget
	m.addConstrs((sum([ workers[j].price*d[i][j] for j in range(num_workers) ]) <= requesters[i].budget for i in range(num_requesters)), 'c4')

	# # c5 means that the a worker may only recieve data shards from one requester
	m.addConstrs(( gurobi.quicksum([ x[i][j] for i in range(num_requesters) ] ) <= 1 for j in range(num_workers)), 'c5')

	# # c6 means that the amount of data shards assigned to a worker must be greater than delta
	m.addConstrs((gurobi.quicksum([ d[i][j] for i in range(num_requesters) ]) >= delta for j in range(num_workers)), 'c6')

	# constrains 7 and 8, that the data assignment must be integral, and x must be binary, is implicit in the variable declarations above

	m.optimize()

	# this is the number of iterations that should be performed for each task
	iterations = [int(e.X) for e in num_iters]

	association = get_2D_list(num_requesters, num_workers)
	allocation = get_2D_list(num_requesters, num_workers)

	for (i, j) in combinations:

		# prints an output
		# if (x[i][j].X == 1):
		# 	print(f'worker: {j} got: {d[i][j].X} data slices from requester {i}')

		association[i][j] = x[i][j].X
		allocation[i][j] = d[i][j].X

	return association, allocation, iterations