import networks
import torch

# the budget per global update iteration
global_budget = 400

tasks = {
	'mnist_ffn': {
		'num_training_iters': 2,
		'num_epochs': 5,
		'deadline': 60,
		# 'deadline': 35,
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
		# 'deadline': 25,
		'network_architecture': networks.ConvNet,
		'dataset_size': 30_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'fashion': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': 45,
		# 'deadline': 30,
		'network_architecture': networks.FashionNet,
		'dataset_size': 10_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'fashion_2': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': 45,
		# 'deadline': 30,
		'network_architecture': networks.FashionNet,
		'dataset_size': 10_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}
}

general_deadline = 40
num_samples = 20_000
tasks_2 = {
	'mnist_cnn_0': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': general_deadline,
		'network_architecture': networks.ConvNet,
		'dataset_size': num_samples // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'mnist_cnn_1': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': general_deadline,
		'network_architecture': networks.ConvNet,
		'dataset_size': num_samples // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'mnist_cnn_2': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': general_deadline,
		'network_architecture': networks.ConvNet,
		'dataset_size': num_samples // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'mnist_cnn_3': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': general_deadline,
		'network_architecture': networks.ConvNet,
		'dataset_size': num_samples // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'mnist_cnn_4': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': general_deadline,
		'network_architecture': networks.ConvNet,
		'dataset_size': num_samples // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'mnist_cnn_5': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': general_deadline,
		'network_architecture': networks.ConvNet,
		'dataset_size': num_samples // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

	'mnist_cnn_6': {
		'num_training_iters': 1,
		'num_epochs': 5,
		'deadline': general_deadline,
		'network_architecture': networks.ConvNet,
		'dataset_size': num_samples // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	},

}