import axon

from MNIST_data import img_x_train, flat_x_train, y_train
from tasks import tasks

data_map = {'mnist_ffn': (flat_x_train, y_train), 'mnist_cnn': (img_x_train, y_train)}

model_map = {}
for task_name in tasks:
	model_map[task_name] = tasks[task_name]['network_architecture']

@axon.worker.rpc()
def get_task_description(task_name):
	return tasks[task_name]

@axon.worker.rpc()
def get_training_data(task_name):
	task_desc = tasks[task_name]
	return data_map[task_desc]

@axon.worker.rpc()
def get_testing_data(task_name):
	pass

axon.worker.rpc()
def clear_params(task_name):
	model_map[task_name] = tasks[task_name]['network_architecture']

@axon.worker.rpc()
def get_parameters(task_name):
	return list(model_map[task_name].parameters())

axon.worker.init()