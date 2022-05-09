import axon
from config import config
import asyncio

async def main():

	learner_ips = axon.discovery.get_ips(ip=config.notice_board_ip)

	print(learner_ips)

	# creates worker handles
	learners = []
	for ip in  learner_ips:
		learners.append(axon.client.RemoteWorker(ip))

	print(await learners[0].rpcs.benchmark('mnist_ffn', 10))

if (__name__ == '__main__'):
	asyncio.run(main())
