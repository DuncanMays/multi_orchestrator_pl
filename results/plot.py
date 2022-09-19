from read_data import avg_across_trials
from matplotlib import pyplot as plt

trial_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
num_workers = [3, 5, 7, 9]

# metric = 'training_time'
metrics = ['resource_util', 'max_training_time', 'training_time', 'EOL', 'acc', 'loss', 'cost', 'sat_ratio', 'total_training_time']

scheme_state_dists = [('MED', 'uncertain'), ('MMTT', 'uncertain'), ('MMTT', 'ideal'), ('TT', 'uncertain'), ('TT', 'ideal')]
# scheme_state_dists = [('TT', 'uncertain'), ('TT', 'ideal'), ('MMTT', 'uncertain'), ('MMTT', 'ideal')]
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

	if (metric == 'acc'):
		plt.ylim((0.7, 0.9))
		plt.legend(prop={'size': 15}, loc='lower left')

	plt.savefig(metric+'_vs_num_workers.png')
	plt.clf()
	# plt.show()