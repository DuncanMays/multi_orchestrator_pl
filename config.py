from types import SimpleNamespace
from torch.cuda import is_available

def set_training_device():
	if is_available():
		return 'cuda:0'
	else:
		return 'cpu'

primative = {
	'data_server_ip': '192.168.2.19',
	'data_server_port': 5002,
	'notice_board_ip': '192.168.2.19',
	'parameter_server_ip': '192.168.2.19',
	'parameter_server_port': 5002,
	'training_device': set_training_device()
}

config = SimpleNamespace(**primative)