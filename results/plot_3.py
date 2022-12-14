from read_data import avg_across_trials
from matplotlib import pyplot as plt
import math

# metrics = ['time_prediction', 'time_prediction_error', 'max_grad_div', 'mean_grad_div', 'resource_util', 'max_training_time', 'training_time', 'acc', 'loss', 'cost', 'worker_sat_ratio', 'task_sat_ratio', 'total_training_time']
# metrics = ['worker_sat_ratio', 'max_training_time', 'resource_util', 'max_grad_div', 'task_fullfillment']
metrics = ['worker_sat_ratio', 'task_fullfillment', 'max_training_time', 'resource_util', 'max_grad_div']
scheme_state_dists = [('MMTT', 'ideal', 'blue'), ('MED', 'uncertain', 'orange'), ('MMTT', 'uncertain', 'green')]
x_labels = [scheme+'_'+state_dist for (scheme, state_dist, _) in scheme_state_dists]
x = [i+1 for i in range(len(x_labels))]

def plot(trial_indices):
	for metric in metrics:

		y = []
		std = []
		for (scheme, state_dist, colour) in scheme_state_dists:
			# remember to change the deadline in tasks.py
			a = avg_across_trials(scheme, state_dist, 'verification', trial_indices)[metric]
			b = avg_across_trials(scheme, state_dist, 'verification_3', trial_indices)[metric]
			c = avg_across_trials(scheme, state_dist, 'verification_5', trial_indices)[metric]

			y.append((a+b+c)/3)
			# std.append(avg_across_trials(scheme, state_dist, 'verification_8', trial_indices)[metric+'_stand_dev'])
		
		# err = [2*s/math.sqrt(len(trial_indices)) for s in std]
		
		y_round = [round(a, 2) for a in y]
		print(metric, y_round)

		# plt.bar(x_labels, y, yerr=err, color=[colour for (_, _, colour) in scheme_state_dists])

		# plt.xlabel('scheme', size=15)
		# plt.yticks(size=10)

		# if (metric == 'worker_sat_ratio'):
		# 	plt.title('Satisfaction Ratio', size=20)
		# 	plt.ylabel('Satisfaction Ratio (%)', size=15)
		# 	plt.ylim(0, 100)

		# if (metric == 'max_training_time'):
		# 	plt.title('Training Time', size=20)
		# 	plt.ylabel('Training Time', size=15)

		# if (metric == 'resource_util'):
		# 	plt.title('Occupancy', size=20)
		# 	plt.ylabel('Occupancy', size=15)
		# 	plt.ylim(0.5, 1)

		# if (metric == 'max_grad_div'):
		# 	plt.title('Parameter Divergence', size=20)
		# 	plt.ylabel('Parameter Divergence', size=15)

		# if (metric == 'task_fullfillment'):
		# 	plt.title('Task Fullfillment', size=20)
		# 	plt.ylabel('Task Fullfillment (%)', size=15)
		# 	plt.ylim(0, 100)

		# plt.savefig('./Nov_28_week/'+metric+'_vs_num_learners.png')
		# plt.show()
		# plt.clf()

# trial_indices = list(range(1, 31))
# trial_indices = list(range(11, 31))
# trial_indices = list(range(1, 21))
# trial_indices = list(range(1, 11)) + list(range(21, 31))


if (__name__ == '__main__'):

	plot(list(range(1, 31)))

	# l = [list(range(1, 11)), list(range(11, 21)), list(range(21, 31))]

	# for trial_indices in l:
	# 	plot(trial_indices)