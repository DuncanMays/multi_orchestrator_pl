# this file exists to help tune the parameters of the stressor functions that put learners in different states
# it doesn't make sense to put the same load on a Jetson Nano as on a raspberry pi. What would moderately inconvenience the Nano would completely cripple the pi
# for this reason, we need to tune stressor parameters with reference to the learner's benchmarking scores

import axon
import asyncio
import sys

sys.path.append('..')
from states import state_dicts
from tasks import tasks

target_ip = '192.168.2.210'

async def main():

	wrkr_handle = axon.client.RemoteWorker(target_ip)

	benchmark_scores = await wrkr_handle.rpcs.get_benchmark_scores()

	# the benchmark scores are unique to a (task, state) pair, the task is irrelavant for our purposes here but we need to pick one
	state_names = list(state_dicts.keys())
	task_names = list(tasks.keys())

	# we only need to see the scores for one task
	task_name = task_names[0]

	print('      | training rate bps | data time spb | param_time spb')

	# iterating over states
	for state_name in state_names:
		ts_key = (task_name, state_name)
		print(state_name, ':', benchmark_scores[ts_key])

if (__name__ == '__main__'):
	asyncio.run(main())