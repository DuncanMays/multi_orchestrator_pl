from utils import get_parameter_server
from config import config
from threading import Thread, Lock

import torch
import time

ps = get_parameter_server()

device = config.training_device

def idle_stressor():
	pass

s = 10*60

a = torch.randn([s, s]).to(device)
b = torch.randn([s, s]).to(device)

def training_stressor(n):
	global a, b

	c = a*b
	b = c*a
	a = b*c

def download_stressor(ps, n):
	a = ps.rpcs.dummy_download.sync_call((10*n, 10*n), {})
	del a

def upload_stressor(ps):
	a = torch.randn([100, 100])
	ps.rpcs.dummy_upload.sync_call((a, ), {})

# this is the dict that describes each state
state_dicts = {
	'idle': {
		'stressor_fn': idle_stressor,
		'params': ( ),
		'probability': 0.2,
	}, 

	'training': {
		'stressor_fn': training_stressor,
		'params': (600, ),
		'probability': 0.4,
	},

	'downloading': {
		'stressor_fn': download_stressor,
		'params': (ps, 5),
		'probability': 0.4,
	},

#	'uploading': {
#		'stressor_fn': upload_stressor,
#		'params': (ps, ),
#		'probability': 0.15,
#	},
}

state = 'idle'
state_lock = Lock()
allowed_states = [state_name for state_name in state_dicts]

def set_state(new_state='idle'):
	global state

	if new_state in allowed_states: 
		with state_lock:
			state = new_state
	
	else:
		Raise(BaseException('invalid state setting'))

def top_level_stressor():
	print('top level stressor has begun')

	# runs a stressor functions once a second based on the state
	while (True):
		time.sleep(1)

		with state_lock:
			stressor_fn = state_dicts[state]['stressor_fn']
			stressor_params = state_dicts[state]['params']

		stressor_fn(*stressor_params)

# this thread runs stressor functions that utilize a certain compute resource to change the learner's compute characteristics
stressor_thread = Thread(target=top_level_stressor)
stressor_thread.daemon = True