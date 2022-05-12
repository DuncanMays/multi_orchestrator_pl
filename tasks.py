import networks
import torch

tasks = {
	'mnist_ffn': {
		'num_training_iters': 1,
		'deadline': 60,
		'budget': 10_000,
		'network_architecture': networks.TwoNN,
		'dataset_name': 'MNIST_flat',
		'dataset_size': 60_000,
		'data_shape': [784],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}, 

	'mnist_cnn': {
		'num_training_iters': 1,
		'deadline': 60,
		'budget': 10_000,
		'network_architecture': networks.ConvNet,
		'dataset_name': 'MNIST_imgs',
		'dataset_size': 60_000,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}
}