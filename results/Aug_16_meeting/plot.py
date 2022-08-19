from read_data import avg_across_trials
from matplotlib import pyplot as plt

trial_indices = list([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
num_workers = [2, 4, 6, 8, 9]

# metric = 'training_time'
metrics = ['training_time', 'EOL', 'acc', 'loss', 'cost', 'sat_ratio']
unit = 's'

scheme_state_dists = [('EOL', 'uncertain'), ('TT', 'uncertain'), ('TT', 'ideal')]
x = [i for i in range(len(num_workers))]

for metric in metrics:

	for (scheme, state_dist) in scheme_state_dists:
		y = [avg_across_trials(scheme, state_dist, str(i)+'_workers', trial_indices)[metric] for i in num_workers]

		plt.plot(x, y, label=scheme+'_'+state_dist)

	plt.title(metric+' versus Number of Workers', size=15)
	plt.ylabel(metric+' ('+unit+')', size=15)
	plt.xlabel('number of workers', size=15)
	plt.xticks(x, num_workers, size=10)
	plt.yticks(size=10)
	plt.legend(prop={'size': 15})
	plt.savefig(metric+'_vs_num_workers.png')
	plt.clf()
	# plt.show()