from types import SimpleNamespace
from torch.cuda import is_available
from tasks import tasks

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
	'default_task_name': next(iter(tasks)),
	'default_deadline': tasks[next(iter(tasks))]['deadline'],
	'default_num_learners': len(tasks),
	'delta': 1,
	'worker_price_mean': 1,
	'worker_price_variance': 0.5,
	'worker_price_max': 1.5,
	'worker_price_min': 0.1,
	'download_stressor_size': 900,
	'training_stressor_size': 900,
	'gaming_stressor_size': 500,
}

config = SimpleNamespace(**config_dict)