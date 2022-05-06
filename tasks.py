import networks

tasks = {
	'mnist_ffn': {
		'num_training_iters': 1,
		'network_architecture': networks.TwoNN,
		'dataset_name': 'MNIST_flat',
		'optimizer': 'Adam',
		'loss': 'CrossEntropy',
	}, 

	'mnist_cnn': {
		'num_training_iters': 1,
		'network_architecture': networks.ConvNet,
		'dataset_name': 'MNIST_imgs',
		'optimizer': 'Adam',
		'loss': 'CrossEntropy',
	}
}