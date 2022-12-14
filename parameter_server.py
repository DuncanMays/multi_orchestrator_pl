import axon
import torch

from data.MNIST_data import img_x_train, flat_x_train, y_train, img_x_test, flat_x_test, y_test
from data.fashion_data import fashion_train_images, fashion_train_labels, fashion_test_images, fashion_test_labels
from tasks import tasks
from utils import set_parameters, average_parameters

BATCH_SIZE = 32

device = 'cpu'
if torch.cuda.is_available():
	device = 'cuda:0'

data_map = {'mnist_ffn': (flat_x_train, y_train), 'mnist_cnn': (img_x_train, y_train), 'fashion': (fashion_train_images, fashion_train_labels)}
test_data_map = {'mnist_ffn': (flat_x_test, y_test), 'mnist_cnn': (img_x_test, y_test), 'fashion': (fashion_test_images, fashion_test_labels)}

# data_map = {'mnist_cnn_0': (img_x_train, y_train), 'mnist_cnn_1': (img_x_train, y_train), 'mnist_cnn_2': (img_x_train, y_train), 'mnist_cnn_3': (img_x_train, y_train), 'mnist_cnn_4': (img_x_train, y_train), 'mnist_cnn_5': (img_x_train, y_train), 'mnist_cnn_6': (img_x_train, y_train),}
# test_data_map = {'mnist_cnn_0': (img_x_test, y_test), 'mnist_cnn_1': (img_x_test, y_test), 'mnist_cnn_2': (img_x_test, y_test), 'mnist_cnn_3': (img_x_test, y_test), 'mnist_cnn_4': (img_x_test, y_test), 'mnist_cnn_5': (img_x_test, y_test), 'mnist_cnn_6': (img_x_test, y_test),}


model_map = {}
for task_name in tasks:
	model_map[task_name] = tasks[task_name]['network_architecture']()

# this map stores lists that hold parameter updates submitted by learners
update_map = {}
for task_name in tasks:
	update_map[task_name] = []

@axon.worker.rpc(executor='Thread')
def get_task_description(task_name):
	return tasks[task_name]

@axon.worker.rpc(executor='Thread')
def get_training_data(task_name, num_shards):
	task_desc = tasks[task_name]
	x_train, y_train = data_map[task_name]

	indices = torch.randint(0, 2, size=[int(num_shards)])

	# it's best to leave the decompression and scaling logic to the client
	x_train, y_train = x_train[indices], y_train[indices]

	return x_train, y_train

@axon.worker.rpc(executor='Thread')
def get_testing_data(task_name, num_shards):
	task_desc = tasks[task_name]
	x_test, y_test = test_data_map[task_name]

	indices = torch.randint(0, 2, size=[int(num_shards)])

	# it's best to leave the decompression and scaling logic to the client
	return x_test[indices], y_test[indices]

@axon.worker.rpc()
def clear_params(task_name):
	# reinstantiates the parameters of the nueral network for a certain task
	model_map[task_name] = tasks[task_name]['network_architecture']()
	# clears the updates for that task
	update_map[task_name] = []

@axon.worker.rpc(executor='Thread')
def get_parameters(task_name):
	P = model_map[task_name].parameters()
	return [p.to('cpu') for p in P]

@axon.worker.rpc(executor='Thread')
def dummy_download(x, y):
	return torch.randn([x, y])

@axon.worker.rpc(executor='Thread')
def dummy_upload(input):
	return input.numel()

@axon.worker.rpc()
def submit_update(task_name, parameters, num_batches):
	global update_map
	print('update submitted for ', task_name, num_batches)
	update_obj = {
		'parameters': parameters,
		'num_batches': num_batches,
	}

	update_map[task_name].append(update_obj)

# returns the 2-norm of 2 tensors of the same shape
def square_distance(a, b):
	c = a-b
	return torch.sum(c*c)

def parameter_norm(a, b):
	param_pairs = [(a[i], b[i]) for i in range(len(a))]
	param_squares = [square_distance(*t) for t in param_pairs]
	return torch.sqrt(sum(param_squares)).item()

@axon.worker.rpc()
def aggregate_parameters(task_name):
	# where updates submitted by learners are stored
	update_objs = update_map[task_name]

	params = [u['parameters'] for u in update_objs]
	num_batches = [u['num_batches'] for u in update_objs]

	print(f'aggregating for: {task_name}')

	# if there haven't been any updates, this will leave the model parameters as they are and prevent the call crashing
	if (len(params) == 0):
		return None, None

	# normalizing weights
	total_batches = sum(num_batches)
	weights = [w/total_batches for w in num_batches]

	aggregate_parameters = average_parameters(params, weights)

	# we now calculate the parameter divergence for each update
	param_divs = []
	for param in params:
		div = parameter_norm(aggregate_parameters, param)
		param_divs.append(div)

	# setting the parameters that are served to learners
	net = model_map[task_name]
	set_parameters(net, aggregate_parameters)

	# clears the update list
	update_map[task_name] = []

	# reports the max and mean parameter divergence back to the caller
	return total_batches, max(param_divs), sum(param_divs)/len(param_divs)

@axon.worker.rpc()
def assess_parameters(task_name, num_shards):
	net = model_map[task_name]
	task_description = tasks[task_name]

	x_shards, y_shards = get_testing_data(task_name, num_shards)

	x_test = x_shards.reshape([x_shards.shape[0]*x_shards.shape[1]]+task_description['data_shape'])/255
	y_test = y_shards.reshape([y_shards.shape[0]*y_shards.shape[1]])

	num_test_batches = x_test.shape[0]//BATCH_SIZE

	loss = 0
	acc = 0

	net = net.to(device)
	criterion = task_description['loss']

	for batch_number in range(num_test_batches):
		x_batch = x_test[BATCH_SIZE*batch_number : BATCH_SIZE*(batch_number+1)].to(device)
		y_batch = y_test[BATCH_SIZE*batch_number : BATCH_SIZE*(batch_number+1)].to(device)

		y_hat = net.forward(x_batch)

		loss += criterion(y_hat, y_batch).item()
		acc += get_accuracy(y_hat, y_batch).item()
	
	# normalizing the loss and accuracy
	loss = loss/num_test_batches
	acc = acc/num_test_batches

	return loss, acc

# calculates the accuracy score of a prediction y_hat and the ground truth y
I = torch.eye(10,10)
def get_accuracy(y_hat, y):
	y_vec = torch.tensor([I[int(i)].tolist() for i in y]).to(device)
	dot = torch.dot(y_hat.flatten(), y_vec.flatten())
	return dot/torch.sum(y_hat)

if (__name__ == '__main__'):
	print('starting parameter server')
	axon.worker.init()