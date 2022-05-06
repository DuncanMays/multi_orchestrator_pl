import axon
import signal
import time

nb_ip = '192.168.2.19'

def shutdown_handler(a, b):
	axon.discovery.sign_out(ip=nb_ip)
	exit()

if (__name__ == '__main__'):

	# registers sign out on sigint
	signal.signal(signal.SIGINT, shutdown_handler)

	# sign into notice board
	axon.discovery.sign_in(ip=nb_ip)

	# starts worker
	print('starting worker')
	time.sleep(60)

	shutdown_handler(None, None)