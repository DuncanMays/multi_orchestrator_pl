import networks
import torch

tasks = {
	'mnist_ffn': {
		'num_training_iters': 1,
		'network_architecture': networks.TwoNN,
		'dataset_name': 'MNIST_flat',
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}, 

	'mnist_cnn': {
		'num_training_iters': 1,
		'network_architecture': networks.ConvNet,
		'dataset_name': 'MNIST_imgs',
		'optimizer': lambda net : torch.optim.Adam([{'params': net.parameters()}], lr=0.0001),
		'loss': torch.nn.CrossEntropyLoss(),
	}
}