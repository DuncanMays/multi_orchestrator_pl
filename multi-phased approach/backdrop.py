import axon
# import parameter_server as ps

class NoticeBoard():

	def __init__(self):
		super(NoticeBoard, self).__init__()
		self.listings = {}
		
	def get_listing(self):
		return list(self.listings.values())

	def sign_in(self, ip_addr, service_name):
		t = (ip_addr, service_name)
		self.listings[hash(t)] = t

	def sign_out(self, ip_addr, service_name):
		t = (ip_addr, service_name)
		try:
			del self.listings[hash(t)]
		except(KeyError):
			raise(BaseException('Not registered'))

if (__name__ == '__main__'):

	nb = NoticeBoard()

	# axon.worker.ServiceNode(ps, 'parameter_server')
	axon.worker.ServiceNode(nb, 'notice_board')

	print('starting backdrop')
	axon.worker.init()
	