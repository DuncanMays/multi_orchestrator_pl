import asyncio
import torch

import learner
from tasks import tasks

def test_get_price():
	print(learner.get_price())

def test_benchmark():
	benchmark_size = 100
	num_downloads = 5

	task_names = [task_name for task_name in tasks]

	for task_name in task_names:
		print(learner.benchmark(task_name, num_downloads, benchmark_size))

def main():
	print('test_get_price')
	test_get_price()

	print('test_benchmark')
	test_benchmark()

if (__name__ == '__main__'):
	# asyncio.run(main())
	main()