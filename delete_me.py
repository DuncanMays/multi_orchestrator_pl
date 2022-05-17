import torch
from tasks import tasks

task_names = [task_name for task_name in tasks]

x = [[1.0, 0.0, 0.0],
	 [0.0, 1.0, 1.0]]

d = [[120.0, 0.0, 0.0],
	 [0.0, 100.0, 20.0]]

# a tensor that holds true on index pairs of workers and requesters who are associated with one another
x = torch.tensor(x) == 1.0

task_indices = x.nonzero()[:, 0].tolist()

association = [task_names[i] for i in task_indices]

# the amount of data allocated to each learner
allocation = torch.tensor(d).sum(dim=0).tolist()

print(allocation)
print(association)