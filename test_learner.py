import asyncio
import torch
import axon

import learner
from tasks import tasks
from config import config

def test_get_price():
	print(learner.get_price())

def test_benchmark():
	benchmark_size = 100
	num_downloads = 5

	task_names = [task_name for task_name in tasks]

	for task_name in task_names:
		print(learner.benchmark(task_name, num_downloads, benchmark_size))

def test_local_update():

	learner.set_training_regime()

	learner.local_update()

def test_training_regime():
	task_name = 'mnist_cnn'
	num_train_shards = 120
	num_epochs = 100

	learner.set_training_regime(task_name, num_train_shards)
	parameter_server = axon.client.RemoteWorker(config.parameter_server_ip)
	parameter_server.rpcs.clear_params.sync_call((task_name, ), {})

	for i in range(num_epochs):
		learner.local_update()
		parameter_server.rpcs.aggregate_parameters.sync_call((task_name, ), {})

def test_startup():
	print(learner.startup())

def main():
	# print(' --- test_get_price --- ')
	# test_get_price()

	# print(' --- test_benchmark --- ')
	# test_benchmark()

	# print(' --- test_local_update --- ')
	# test_local_update()

	# print(' --- test_training_regime --- ')
	# test_training_regime()

	print(' --- test_startup --- ')
	test_startup()

if (__name__ == '__main__'):
	# asyncio.run(main())
	main()