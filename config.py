from types import SimpleNamespace
from torch.cuda import is_available

def get_training_device():
	if is_available():
		return 'cuda:0'
	else:
		return 'cpu'

config_dict = {
	'data_server_ip': '192.168.2.19',
	'data_server_port': 5002,
	'notice_board_ip': '192.168.2.19',
	'parameter_server_ip': '192.168.2.19',
	'parameter_server_port': 5002,
	'training_device': get_training_device(),
	'default_task_name': 'mnist_ffn',
	'delta': 1
}

config = SimpleNamespace(**config_dict)