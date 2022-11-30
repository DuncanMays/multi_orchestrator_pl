from read_data import avg_across_trials
from matplotlib import pyplot as plt
import math

trial_indices = list(range(1, 31))
# trial_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# num_learners = [3, 5, 7, 9]
num_learners = [5, 7, 9, 11, 13]

# metrics = ['time_prediction', 'time_prediction_error', 'max_grad_div', 'mean_grad_div', 'resource_util', 'max_training_time', 'training_time', 'acc', 'loss', 'cost', 'worker_sat_ratio', 'task_sat_ratio', 'total_training_time']
metrics = ['worker_sat_ratio', 'max_training_time', 'resource_util', 'max_grad_div']

scheme_state_dists = [('MMTT', 'ideal', 'blue'), ('MMTT', 'static', 'red'), ('MED', 'uncertain', 'orange'), ('MMTT', 'uncertain', 'green')]
x = [i for i in range(len(num_learners))]

for metric in metrics:

	# if (metric == 'acc'):
	# acc should be a bar graph from 0 to 100
	offset = -0.2
	step = 0.2
	for (scheme, state_dist, colour) in scheme_state_dists:
		y = [avg_across_trials(scheme, state_dist, str(i)+'_num_learners', trial_indices)[metric] for i in num_learners]
		std = [avg_across_trials(scheme, state_dist, str(i)+'_num_learners', trial_indices)[metric+'_stand_dev'] for i in num_learners]
		err = [2*s/math.sqrt(len(trial_indices)) for s in std]

		plt.bar([a + offset for a in x], y, yerr=err, width=0.2, label=scheme+'_'+state_dist, color=colour)
		# plt.errorbar([a + offset for a in x], y, err, mfc='red', label=scheme+'_'+state_dist)

		offset += step


	plt.xlabel('num_learners', size=15)
	plt.xticks(x, num_learners, size=10)
	plt.yticks(size=10)
	plt.legend(prop={'size': 15})

	if (metric == 'worker_sat_ratio'):
		plt.title('Satisfaction Ratio', size=20)
		plt.ylabel('Satisfaction Ratio', size=15)
		plt.ylim(0.8, 1)
		plt.legend(prop={'size': 15}, loc='lower left')

	if (metric == 'max_training_time'):
		plt.title('Training Time', size=20)
		plt.ylabel('Training Time', size=15)
		plt.ylim(0, 20)
		plt.legend(prop={'size': 15}, loc='lower left')

	if (metric == 'resource_util'):
		plt.title('Occupancy', size=20)
		plt.ylabel('Occupancy', size=15)
		plt.ylim(0.6, 1)
		plt.legend(prop={'size': 15}, loc='lower left')

	if (metric == 'max_grad_div'):
		plt.title('Parameter Divergence', size=20)
		plt.ylabel('Parameter Divergence', size=15)
		plt.legend(prop={'size': 15}, loc='lower right')

	# plt.savefig('./Nov_28_week/'+metric+'_vs_num_learners.png')
	plt.show()
	plt.clf()