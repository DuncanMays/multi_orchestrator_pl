from read_data import avg_across_trials
from matplotlib import pyplot as plt

trial_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
heat_params = [0.0001, 0.5, 1.0, 1.5, 2.0]

metrics = ['resource_util', 'max_training_time', 'training_time', 'EOL', 'acc', 'loss', 'cost', 'worker_sat_ratio', 'task_sat_ratio', 'total_training_time']

scheme_state_dists = [('MMTT', 'ideal'), ('MED', 'uncertain'), ('MMTT', 'uncertain')]
x = [i for i in range(len(heat_params))]

for metric in metrics:

	if (metric == 'acc'):
		# acc should be a bar graph from 0 to 100
		offset = -0.2
		step = 0.2
		for (scheme, state_dist) in scheme_state_dists:
			y = [avg_across_trials(scheme, state_dist, str(i)+'_heat_param', trial_indices)[metric] for i in heat_params]
			plt.bar([a + offset for a in x], y, width=0.2, label=scheme+'_'+state_dist)
			offset += step

	else:
		for (scheme, state_dist) in scheme_state_dists:
			y = [avg_across_trials(scheme, state_dist, str(i)+'_heat_param', trial_indices)[metric] for i in heat_params]

			plt.plot(x, y, label=scheme+'_'+state_dist)

	if (metric == 'resource_util'):
		plt.title('Worker Occupation Time versus State Heat', size=15)
		plt.ylabel('Worker Occupation Time', size=15)

	plt.xlabel('state heat', size=15)
	plt.xticks(x, heat_params, size=10)
	plt.yticks(size=10)
	plt.legend(prop={'size': 15})

	if (metric == 'task_sat_ratio'):
		plt.legend(prop={'size': 15}, loc='upper right')

	if (metric == 'max_training_time'):
		plt.legend(prop={'size': 15}, loc='lower right')

	if (metric == 'acc'):
		plt.ylim(0, 1)
		plt.legend(prop={'size': 15}, loc='lower left')

	plt.savefig(metric+'_vs_state_heat.png')
	# plt.show()
	plt.clf()