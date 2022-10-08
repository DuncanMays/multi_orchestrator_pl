from config import config
from threading import Thread, Lock

import torch
import time
import random

device = config.training_device

def idle_stressor():
	pass

def training_stressor(s):
	a = torch.randn([s, s]).to(device)
	b = torch.randn([s, s]).to(device)
	
	c = a*b
	b = c*a
	a = b*c

def download_stressor(ps, n):
	for _ in range(n):
		ps.rpcs.dummy_download.sync_call((5000, 5000), {})

# this is the dict that describes each state
state_dicts = {
	'idle': {
		'stressor_fn': idle_stressor,
		# 'params': ( ),
		# 'probability': 0.34
	}, 

	'training': {
		'stressor_fn': training_stressor,
		# 'params': (50, ),
		# 'probability': 0.33
	},

	'downloading': {
		'stressor_fn': download_stressor,
		# 'params': (ps, 50),
		# 'probability': 0.33
	},

}

state = 'idle'
state_lock = Lock()
allowed_states = [state_name for state_name in state_dicts]

# state_heat controls how spread out the worker's state distribution is, hotter distributions are more spread out, colder ones more concentrated on one state
state_heat = 1/2
# this is the distribution that controls how workers change states
state_distribution = torch.softmax(torch.randint(0, 2, (len(allowed_states), ))/state_heat, dim=0).tolist()

def get_state():
	global state
	with state_lock:
		return state

def get_state_distribution():
	global state_distribution
	return state_distribution

def set_state_distribution(new_distribution):
	global state_distribution

	if (sum(new_distribution) >= 1.01) or (sum(new_distribution) <= 0.99):
		raise BaseException('invalid probability distribution')

	if (len(new_distribution) != len(allowed_states)):
		raise BaseException('number of indices in distribution list doesn\'t match number of states')

	state_distribution = new_distribution

# this function sets the state of the worker, if no input is given it samples an input from the worker's state distribution
def set_state(new_state=None):
	global state

	if (new_state == None):
		new_state = random.choices(allowed_states, state_distribution).pop()

		with state_lock:
			state = new_state

	elif new_state in allowed_states: 
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
