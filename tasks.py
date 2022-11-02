import networks
import torch

global_budget = 300

tasks = {
	'mnist_ffn': {
		'num_training_iters': 2,
		'num_epochs': 5,
		# 'deadline': 50,
		'deadline': 30,
		'network_architecture': networks.ThreeNN,
		'dataset_size': 30_000 // 500,
		'data_shape': [784],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}, 

	'mnist_cnn': {
		'num_training_iters': 1,
		'num_epochs': 5,
		# 'deadline': 45,
		'deadline': 25,
		'network_architecture': networks.ConvNet,
		'dataset_size': 30_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'fashion': {
		'num_training_iters': 1,
		'num_epochs': 5,
		# 'deadline': 60,
		'deadline': 40,
		'network_architecture': networks.FashionNet,
		'dataset_size': 10_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}
}

tasks_1 = {
	'mnist_cnn_0': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': 45,
		'network_architecture': networks.ConvNet,
		'dataset_size': 30_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'mnist_cnn_1': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': 45,
		'network_architecture': networks.ConvNet,
		'dataset_size': 30_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'mnist_cnn_2': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': 45,
		'network_architecture': networks.ConvNet,
		'dataset_size': 30_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'mnist_cnn_3': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': 45,
		'network_architecture': networks.ConvNet,
		'dataset_size': 30_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'mnist_cnn_4': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': 45,
		'network_architecture': networks.ConvNet,
		'dataset_size': 30_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

}