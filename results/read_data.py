import json
import sys
import os

sys.path.append('..')
from tasks import tasks

target_dir = 'Sep_1_meeting'

metrics = ['max_training_time', 'loss', 'acc', 'EOL', 'cost', 'total_training_time', 'training_time', 'sat_ratio']
task_names = list(tasks.keys())

def get_empty_data_point():
	return {metric: 0 for metric in metrics}

def get_acc_data(scheme, state_distribution, experiment_name, trial_number, task_name):
	filename = scheme+'_'+state_distribution+'_'+experiment_name+'_'+str(trial_number)+'.json'
	filepath = os.path.join('.', target_dir, filename)

	with open(filepath, 'r') as f:
		raw_data = json.loads(f.read())

		return [d['acc'] for d in raw_data['training_metrics'][task_name]]


# Given a scheme (EOL vs TT), a state distribution (ideal/uncertain), experiment name and trial number, report the cost, EOL, final loss and acc, as well as the average training time and sat ratio
def get_data(scheme, state_distribution, experiment_name, trial_number):

	filename = scheme+'_'+state_distribution+'_'+experiment_name+'_'+str(trial_number)+'.json'
	filepath = os.path.join('.', target_dir, filename)

	with open(filepath, 'r') as f:
		raw_data = json.loads(f.read())

	data_point = get_empty_data_point()

	# EOL and cost are easy to obtain
	data_point['EOL'] = raw_data['EOL']
	data_point['cost'] = raw_data['cost']

	# averaging converged loss and acc accross tasks
	data_point['loss'] = sum([raw_data['training_metrics'][task_name][-1]['loss'] for task_name in task_names])/len(task_names)
	data_point['acc'] = sum([raw_data['training_metrics'][task_name][-1]['acc'] for task_name in task_names])/len(task_names)

	avg_training_time = 0
	avg_max_training_time = -1
	avg_sat_ratio = 0
	avg_total_training_time = 0

	for task_name in task_names:
		# ditching the first index as that has training time zero
		# training times is a list of lists, the first dimension moves accross global update cycles, the second dimension accross learners
		training_times = [d['times'] for d in raw_data['training_metrics'][task_name][1:]]

		# if (task_name == 'fashion'):
		# 	print([max(t) for t in training_times])

		# the average training time for each worker
		avg_times = [sum(t)/len(t) for t in training_times]
		avg_training_time += sum(avg_times)/len(avg_times)

		# the max training time of each worker in a global update, this is the time for the globa update
		max_times = [max(t) for t in training_times]
		avg_max_training_time += sum(max_times)/len(max_times)

		# this metric gives the total amount of time until convergence
		# calculated as the sum of the amount of time for the longest running learner in each global update cycle
		total_training_time = sum([max(t) for t in training_times])
		avg_total_training_time += total_training_time

		# the satisfaction ration is the number of workers who returned before the deadline
		# satisfied_deadlines = sum([sum([t < tasks[task_name]['deadline'] for t in T]) for T in training_times])
		# total_deadlines = sum([len(t) for t in training_times])

		# adds a 1 or 0 depending if the task finished before the deadline this trial
		if (total_training_time < len(training_times)*tasks[task_name]['deadline']): 
			avg_sat_ratio += 1

	data_point['training_time'] = avg_training_time/len(tasks)
	data_point['max_training_time'] = avg_max_training_time/len(tasks)
	data_point['total_training_time'] = avg_total_training_time/len(tasks)
	data_point['sat_ratio'] = avg_sat_ratio/len(tasks)

	return data_point

def avg_across_trials(scheme, state_distribution, experiment_name, trial_indices):
	avg_data = get_empty_data_point()
	accs = []

	for i in trial_indices:
		d = get_data(scheme, state_distribution, experiment_name, i)

		accs.append(d['acc'])

		for metric in metrics:
			avg_data[metric] += d[metric]

	for metric in metrics:
		avg_data[metric] = avg_data[metric]/len(trial_indices)

	# if (experiment_name == '5_workers'):
	# 	print(scheme)
	# 	print(accs)

	return avg_data