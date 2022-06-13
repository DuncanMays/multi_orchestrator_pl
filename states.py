from utils import get_parameter_server
from config import config
import torch

ps = get_parameter_server()

# device = config.training_device
device = 'cpu'

def idle_stressor():
	pass

def training_stressor():

	a = torch.randn([100, 100]).to(device)
	b = torch.randn([100, 100]).to(device)

	c = a*b
	b = c*a
	a = b*c

def download_stressor(ps):
	ps.rpcs.dummy_download.sync_call((100, 100), {})

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
		'params': ( ),
		'probability': 0.4,
	},

	'downloading': {
		'stressor_fn': download_stressor,
		'params': (ps, ),
		'probability': 0.25,
	},

	'uploading': {
		'stressor_fn': upload_stressor,
		'params': (ps, ),
		'probability': 0.15,
	},
}