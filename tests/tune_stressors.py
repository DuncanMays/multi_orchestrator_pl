# this file exists to help tune the parameters of the stressor functions that put learners in different states
# it doesn't make sense to put the same load on a Jetson Nano as on a raspberry pi. What would moderately inconvenience the Nano would completely cripple the pi
# for this reason, we need to tune stressor parameters with reference to the learner's benchmarking scores

import axon
import asyncio

from states import state_dicts

target_ip = '192.168.2.210'

async def main():

	wrkr_handle = axon.client.RemoteWorker(target_ip)

	# runs benchmarks in worker
	await wrkr_handle.rpcs.startup()

	benchmark_scores = await wrkr_handle.rpcs.get_benchmark_scores()

	# the benchmark scores are unique to a (task, state) pair, the task is irrelavant for our purposes here but we need to pick one
	task_index = 0
	state_names = list(state_dicts.keys())

	print('      | training rate bps | data time spb | param_time spb')

	# iterating over states
	for i in range(len(state_dicts)):
		ts = (task_index, i)

		print(state_names[i], ':', benchmark_scores[ts])

if (__name__ == '__main__'):
	asyncio.run(main())