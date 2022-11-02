from matplotlib import pyplot as plt
import json
import os
import sys
sys.path.append('..')
from tasks import tasks_0 as tasks
task_names = tasks.keys()

scheme_state_dists = [('MMTT', 'ideal'), ('MED', 'uncertain'), ('MMTT', 'uncertain')]

def read_data(scheme, state_dist, experiment_name, trial_index):

	target_dir = './Oct_28_meeting/allocation_data'
	# file_name = 'MMTT_uncertain_3_record_allocation_5.json'
	file_name = scheme+'_'+state_dist+'_'+experiment_name+'_'+str(trial_index)+'.json'
	filepath = os.path.join(target_dir, file_name)

	data = None
	with open(filepath, 'r') as f:
		data = json.loads(f.read())

	return data

spread = lambda l : max(l) - min(l)

def average_over_trials(scheme, state_dist, experiment_name, trial_indices):
	metric_fn = spread
	avg_metric_dict = {name : 0 for name in task_names}

	for t in trial_indices:
		data = read_data(scheme, state_dist, experiment_name, t)

		for task_name in task_names:
			avg_metric_dict[task_name] += metric_fn(data[task_name])

	for task_name in task_names:
		avg_metric_dict[task_name] = avg_metric_dict[task_name]/len(trial_indices)

	return avg_metric_dict

def print_data():
	for scheme, state_dist in scheme_state_dists:
		avg_data = average_over_trials(scheme, state_dist, '3_record_allocation', list(range(1, 31)))
		avg_metric = 0
		for t in task_names:
			avg_metric += avg_data[t]/len(task_names)	

		print(avg_metric)

def plot_data(data):
	x = [i for i in range(len(data))]

	plt.bar(x, data, color=['blue', 'orange', 'green'])

	plt.title('Allocation Spread', size=15)
	plt.xticks(x, ['MMTT_ideal', 'MED', 'MMTT_uncertain'], size=15)

	plt.show()

if (__name__ == '__main__'):
	# print_data()
	data = [2.3888888888888884, 6.666666666666666, 9.644444444444444]
	plot_data(data)