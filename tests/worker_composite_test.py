import sys

sys.path.append('..')

from worker_composite import gather_wrapper, WorkerComposite
import asyncio
from random import randint
import axon
from threading import Thread

# helper function for test_gather_wrapper
async def print_hello(x=None):
	a = randint(0, 2)
	await asyncio.sleep(a)
	print('hello', x)

async def test_gather_wrapper():
	num_coros = 10

	child_coros = [print_hello for i in range(num_coros)]
	wrapped = gather_wrapper(child_coros)
	print(await wrapped([('there', ) for i in range(num_coros)]))

async def test_WorkerComposite():
	# we start a dummy worker in a separate thread
	@axon.worker.rpc()
	def hello_there():
		print('hello there!')

	worker_thread = Thread(target=axon.worker.init)
	worker_thread.daemon = True
	worker_thread.start()

	# we now create multiple worker handles and gather them into a composite class 
	await asyncio.sleep(0.2)

	num_workers = 5
	wrkr_handles = [axon.client.RemoteWorker('localhost') for i in range(num_workers)]

	wc = WorkerComposite(wrkr_handles)
	await wc.rpcs.hello_there()

async def main():

	# await test_gather_wrapper()

	await test_WorkerComposite()

if __name__ == '__main__':
	asyncio.run(main(), debug=True)