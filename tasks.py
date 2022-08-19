import networks
import torch

global_budget = 120

tasks = {
	'mnist_ffn': {
		'num_training_iters': 2,
		'num_epochs': 5,
		'deadline': 35,
		'network_architecture': networks.ThreeNN,
		'dataset_size': 20_000 // 500,
		'data_shape': [784],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}, 

	'mnist_cnn': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': 30,
		'network_architecture': networks.ConvNet,
		'dataset_size': 20_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	# 'fashion': {
	# 	'num_training_iters': 1,
	# 	'num_epochs': 5,
	# 	'deadline': 400,
	# 	'budget':800,
	# 	'network_architecture': networks.FashionNet,
	# 	'dataset_name': 'MNIST_imgs',
	# 	'dataset_size': 2000 // 500,
	# 	'data_shape': [1, 28, 28],
	# 	'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
	# 	'loss': torch.nn.CrossEntropyLoss(),
	# }
}