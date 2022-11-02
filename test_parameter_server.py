import parameter_server as ps
from networks import ThreeNN, TwoNN
from utils import set_parameters
from tqdm import tqdm
import torch

# instantiates network with parameters that serve as the main model prior to being sent to workers
main_net = TwoNN()
main_parameters = list(main_net.parameters())

if (__name__ == '__main__'):
	num_updates = 5
	training_iters = 500

	for i in range(num_updates):

		net = TwoNN()
		update_weight = 0
		set_parameters(net, main_parameters)

		if (i == 0):
		# if True:
			print('training a little')

			optim = torch.optim.Adam([{'params': net.parameters()}], lr=0.001)
			criteria = torch.nn.MSELoss()

			y = torch.zeros([32, 10], dtype=torch.float32)
			y[:, 0] = float(1)

			update_weight += training_iters

			for j in tqdm(range(training_iters)):
				x = torch.randn([32, 784], dtype=torch.float32)

				y_hat = net(x)
				loss = criteria(y, y_hat)

				optim.zero_grad()
				loss.backward()
				optim.step()

				# print(loss.item())

		# creates a parameter udpate
		param_update = [p.to('cpu') for p in list(net.parameters())]

		# uploads to PS
		ps.submit_update('mnist_ffn', param_update, update_weight)

	print(ps.aggregate_parameters('mnist_ffn'))