import axon
from config import config

parameter_server = None
def get_parameter_server():
	global parameter_server

	if (parameter_server == None):
		parameter_server = axon.client.RemoteWorker(config.parameter_server_ip)

	return parameter_server

# sets the parameters of a neural net
def set_parameters(net, params):
	current_params = list(net.parameters())
	for i, p in enumerate(params): 
		current_params[i].data = p.data.clone()

# this function averages parameters from workers, according to a weight
def average_parameters(param_list, weights):

	avg_params = []

	for i, params in enumerate(param_list):

		if (i == 0):
			for p in params:
				avg_params.append(p.clone()*weights[i])

		else:
			for j, p in enumerate(params):
				avg_params[j].data += p.data*weights[i]

	return avg_params