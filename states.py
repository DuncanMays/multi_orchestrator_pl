from utils import get_parameter_server
from config import config
import torch

ps = get_parameter_server()

# device = config.training_device
device = 'cpu'

def idle_stressor():
	pass

def training_stressor(n):
	s = 10*n

	a = torch.randn([s, s]).to(device)
	b = torch.randn([s, s]).to(device)

	c = a*b
	b = c*a
	a = b*c

def download_stressor(ps, n):
	ps.rpcs.dummy_download.sync_call((10*n, 10*n), {})

def upload_stressor(ps):
	a = torch.randn([100, 100])
	ps.rpcs.dummy_upload.sync_call((a, ), {})

state_dicts = {
	'idle': {
		'stressor_fn': idle_stressor,
		'params': ( ),
		'probability': 0.2,
	}, 

	'training': {
		'stressor_fn': training_stressor,
		'params': (800, ),
		'probability': 0.4,
	},

	'downloading': {
		'stressor_fn': download_stressor,
		'params': (ps, 30),
		'probability': 0.4,
	},

#	'uploading': {
#		'stressor_fn': upload_stressor,
#		'params': (ps, ),
#		'probability': 0.15,
#	},
}
