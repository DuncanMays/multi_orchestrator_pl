import sys

sys.path.append('..')
import parameter_server
from tasks import tasks

def test_get_training_data():
	x_data, y_data = parameter_server.get_training_data('mnist_ffn', 50)
	print(x_data.shape)

def test_submit_update_and_aggregate():
	task_name = 'mnist_ffn'
	net = tasks[task_name]['network_architecture']()

	parameter_server.submit_update(task_name, list(net.parameters()), 50)
	parameter_server.submit_update(task_name, list(net.parameters()), 50)
	parameter_server.submit_update(task_name, list(net.parameters()), 50)

	parameter_server.aggregate_parameters(task_name)
	aggregated_parameters = parameter_server.get_parameters(task_name)

	print(list(aggregated_parameters)[0] == list(net.parameters())[0])

def test_assess_parameters():
	task_name = 'mnist_ffn'
	num_shards = 100

	print('loss and acc:', parameter_server.assess_parameters(task_name, 100))

def main():
	# print('test_get_training_data')
	# test_get_training_data()

	# print('test_submit_update_and_aggregate')
	# test_submit_update_and_aggregate()
	
	print('test_assess_parameters')
	test_assess_parameters()

if (__name__ == '__main__'):
	main()
	