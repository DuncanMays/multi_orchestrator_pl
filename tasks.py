import networks
import torch

global_budget = 155

tasks = {
	'mnist_ffn': {
		'num_training_iters': 2,
		'num_epochs': 5,
		'deadline': 50,
		'network_architecture': networks.ThreeNN,
		'dataset_size': 30_000 // 500,
		'data_shape': [784],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}, 

	'mnist_cnn': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': 45,
		'network_architecture': networks.ConvNet,
		'dataset_size': 30_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'fashion': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': 60,
		'network_architecture': networks.FashionNet,
		'dataset_size': 10_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}
}