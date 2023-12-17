import axon
from config import config

parameter_server = None
def get_parameter_server():
	global parameter_server

	if (parameter_server == None):
		parameter_server = axon.client.get_RemoteWorker(config.parameter_server_ip)

	return parameter_server

# sets the parameters of a neural net
def set_parameters(net, params):
	current_params = list(net.parameters())
	for i, p in enumerate(params): 
		current_params[i].data = p.data.clone()

# param_list is a list of lists of tensors
def average_parameters(param_list, weights):

	# the length of each list of parameter tensors in param_list
	param_len = len(param_list[0])
	averaged_params = []

	for p_index in range(param_len):
		param_tensors = [p[p_index] for p in param_list]
		weighted_param_tensors = [p*weights[i] for i, p in enumerate(param_tensors)]
		averaged_p = sum(weighted_param_tensors)
		averaged_params.append(averaged_p)

	return averaged_params