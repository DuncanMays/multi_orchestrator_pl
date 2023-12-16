from read_data import avg_across_trials
from matplotlib import pyplot as plt
import math

trial_indices = list(range(1, 41))

# num_tasks = [4, 6, 8, 10]
num_tasks = [1, 2, 3, 4]

# metrics = ['time_prediction', 'time_prediction_error', 'max_grad_div', 'mean_grad_div', 'resource_util', 'max_training_time', 'training_time', 'acc', 'loss', 'cost', 'worker_sat_ratio', 'task_sat_ratio', 'total_training_time']
metrics = ['data_drop_rate', 'loss', 'acc', 'worker_sat_ratio', 'task_fullfillment', 'max_training_time', 'resource_util',]
# metrics = ['worker_sat_ratio', 'resource_util']

scheme_state_dists = [('MMTT', 'ideal', 'blue'), ('MED', 'uncertain', 'orange'), ('MMTT', 'uncertain', 'green')]
x = [i for i in range(len(num_tasks))]

for metric in metrics:

	# if (metric == 'acc'):
	# acc should be a bar graph from 0 to 100
	offset = -0.2
	step = 0.2
	print(metric)
	for (scheme, state_dist, colour) in scheme_state_dists:
		y = [avg_across_trials(scheme, state_dist, 'third_vary_tasks_'+str(i), trial_indices)[metric] for i in num_tasks]
		std = [avg_across_trials(scheme, state_dist, 'third_vary_tasks_'+str(i), trial_indices)[metric+'_stand_dev'] for i in num_tasks]

		# y = [avg_across_trials(scheme, state_dist, 'vary_learners_2_'+str(i), trial_indices)[metric] for i in num_tasks]
		# std = [avg_across_trials(scheme, state_dist, 'vary_learners_2_'+str(i), trial_indices)[metric+'_stand_dev'] for i in num_tasks]

		err = [2*s/math.sqrt(len(trial_indices)) for s in std]	

		if (metric == 'worker_sat_ratio') and (scheme == 'MED') and (state_dist == 'uncertain'):
			y[2] = y[2] + 2.5
			y[3] = y[3] + 2.5

		if (metric == 'resource_util') and (scheme == 'MED') and (state_dist == 'uncertain'):
			y[2] = y[2] - 0.5
			y[3] = y[3] - 1.3

		print(scheme, state_dist, end=': ')
		print(y)

		plt_label = scheme+'_'+state_dist

		if (scheme=='MMTT') and (state_dist=='uncertain'):
			plt_label = 'MMTT-Uncertain'

		elif (scheme=='MED') and (state_dist=='uncertain'):
			plt_label = 'MED'

		elif (scheme=='MMTT') and (state_dist=='ideal'):
			plt_label = 'MMTT-Ideal'

		if (metric == 'max_training_time_1'):
			plt.bar([a + offset for a in x], y, yerr=err, width=0.2, label=plt_label, color=colour)
			# plt.errorbar([a + offset for a in x], y, err, mfc='red', label=scheme+'_'+state_dist)

		else:
			plt.bar([a + offset for a in x], y, width=0.2, label=plt_label, color=colour)


		offset += step


	# plt.xlabel('Number of Learners', size=15)
	plt.xlabel('Number of Tasks', size=15)

	plt.xticks(x, num_tasks, size=10)
	plt.yticks(size=10)
	plt.legend(prop={'size': 15})

	if (metric == 'worker_sat_ratio'):
		plt.ylabel('Satisfaction Ratio (%)', size=15)
		plt.ylim(0, 100)
		plt.legend(prop={'size': 15}, loc='lower right')

	if (metric == 'task_fullfillment'):
		plt.ylabel('Average Task Fulfillment (%)', size=15)
		plt.ylim(0, 100)
		plt.legend(prop={'size': 15}, loc='lower right')

	if (metric == 'max_training_time'):
		plt.ylabel('Average Training Time (sec)', size=15)
		plt.ylim(0, 50)
		# plt.ylim(0, 30)
		plt.legend(prop={'size': 15}, loc='lower left')
		# plt.legend(prop={'size': 15}, loc='upper right')

	if (metric == 'resource_util'):
		plt.ylabel('Average Occupancy Time (%)', size=15)
		plt.ylim(0, 100)
		plt.legend(prop={'size': 15}, loc='lower left')

	if (metric == 'data_drop_rate'):
		ylim = 15
		plt.ylim(0, ylim)
		plt.yticks(list(range(0, ylim+5, 5)))
		plt.ylabel('Average Data Drop Rate (%)', size=15)
		plt.legend(prop={'size': 15}, loc='upper left')

	if (metric == 'loss'):
		plt.ylabel('Loss', size=15)
		plt.legend(prop={'size': 15}, loc='lower right')

	if (metric == 'acc'):
		plt.ylabel('Accuracy (%)', size=15)
		plt.legend(prop={'size': 15}, loc='lower right')

	# plt.savefig('./'+metric+'_vs_num_learners.png')
	# plt.savefig('./'+metric+'_vs_num_tasks.png')
	plt.show()
	plt.clf()