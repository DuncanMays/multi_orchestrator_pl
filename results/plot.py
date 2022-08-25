from read_data import avg_across_trials
from matplotlib import pyplot as plt

trial_indices = list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
num_workers = [3, 5, 7, 9]

# metric = 'training_time'
metrics = ['training_time', 'EOL', 'acc', 'loss', 'cost', 'sat_ratio']

scheme_state_dists = [('EOL', 'uncertain'), ('EOL_max', 'uncertain'), ('TT', 'uncertain'), ('TT', 'ideal')]
x = [i for i in range(len(num_workers))]

for metric in metrics:

	for (scheme, state_dist) in scheme_state_dists:
		y = [avg_across_trials(scheme, state_dist, str(i)+'_workers', trial_indices)[metric] for i in num_workers]

		# print(y)

		plt.plot(x, y, label=scheme+'_'+state_dist)

	plt.title(metric+' versus Number of Workers', size=15)
	plt.ylabel(metric, size=15)
	plt.xlabel('number of workers', size=15)
	plt.xticks(x, num_workers, size=10)
	plt.yticks(size=10)
	plt.legend(prop={'size': 15})
	plt.savefig(metric+'_vs_num_workers.png')
	plt.clf()
	# plt.show()