import torch
import random

names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
shard_size = 5

# reading data from file
x_data = []
y_data = []

raw_data = None
with open('./iris.data', 'r') as f:
	raw_data = f.read()

raw_lines = raw_data.split('\n')
# the last two lines are non data
del raw_lines[-1], raw_lines[-1]

# we now format the data into input arrays and name indixes
for line in raw_lines:
	# columns are deliminated by commas
	features = line.split(',')
	# the last column is the name
	name = features.pop()

	# turns features from strings to floats
	feature_floats = [float(f) for f in features]
	# gets an integer representation of the name
	name_index = names.index(name)

	x_data.append(feature_floats)
	y_data.append(name_index)

x_tensor = torch.tensor(x_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.long)

# normalizing input along data sample axis
# getting the max and min value
x_max, _ = x_tensor.max(dim=0)
x_min, _ = x_tensor.min(dim=0)
# expanding the max and min tensors into the desired shape
x_max = x_max.unsqueeze(dim=0).repeat([x_tensor.shape[0], 1])
x_min = x_min.unsqueeze(dim=0).repeat([x_tensor.shape[0], 1])
# scaling x_tensor to between 0 and 1
x_tensor = (x_tensor - x_min) / (x_max - x_min)
del x_max, x_min

# we now shuffle data
indices = list(range(x_tensor.shape[0]))
random.shuffle(indices)
x_tensor = x_tensor[indices]
y_tensor = y_tensor[indices]

# splitting into shards
num_shards = x_tensor.shape[0] // shard_size

# cutting off the samples that won't be in any shard
x_tensor = x_tensor[0: num_shards*shard_size]
y_tensor = y_tensor[0: num_shards*shard_size]

x_train = x_tensor.unsqueeze(dim=0).reshape([num_shards, -1, 4])
y_train = y_tensor.unsqueeze(dim=0).reshape([num_shards, -1])