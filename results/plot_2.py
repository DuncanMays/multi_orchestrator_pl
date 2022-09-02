from read_data import avg_across_trials, get_acc_data
from matplotlib import pyplot as plt

trial_indices = list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

for i in trial_indices:
	y = get_acc_data('EOL_max', 'uncertain', '7_workers', i, 'mnist_ffn')
	x = [i for i in range(len(y))]

	plt.plot(x, y)

plt.show()
