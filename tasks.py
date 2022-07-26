import networks
import torch

tasks = {
	'mnist_ffn': {
		'num_training_iters': 2,
		'deadline': 30,
		'budget': 150_000,
		'network_architecture': networks.ThreeNN,
		'dataset_name': 'MNIST_flat',
		'dataset_size': 20_000 // 500,
		'data_shape': [784],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}, 

	'mnist_cnn': {
		'num_training_iters': 1,
		'deadline': 30,
		'budget': 100_000,
		'network_architecture': networks.ConvNet,
		'dataset_name': 'MNIST_imgs',
		'dataset_size': 20_000 // 500,
		'data_shape': [1, 28, 28],
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}
}