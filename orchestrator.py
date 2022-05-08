import axon
from config import config

if (__name__ == '__main__'):

	learner_ips = axon.discovery.get_ips(ip=config.notice_board_ip)

	print(learner_ips)