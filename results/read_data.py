import json
import sys
import os

sys.path.append('..')
from tasks import tasks

target_dir = 'Aug_25_meeting/data_files'

metrics = ['loss', 'acc', 'EOL', 'cost', 'training_time', 'sat_ratio']
task_names = list(tasks.keys())

def get_empty_data_point():
	return {metric: 0 for metric in metrics}

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

	# averaging loss and acc accross tasks
	data_point['loss'] = sum([raw_data['training_metrics'][task_name][-1]['loss'] for task_name in task_names])/len(task_names)
	data_point['acc'] = sum([raw_data['training_metrics'][task_name][-1]['acc'] for task_name in task_names])/len(task_names)

	avg_training_time = 0
	avg_sat_ratio = 0
	for task_name in task_names:
		# ditching the first index as that has training time zero
		training_times = [d['times'][0] for d in raw_data['training_metrics'][task_name][1:]]
		avg_training_time += sum(training_times)/len(training_times)

		deadline_satisfaction = [sum([d['times'][0] < tasks[task_name]['deadline']]) for d in raw_data['training_metrics'][task_name][1:]]
		avg_sat_ratio += sum(deadline_satisfaction)/len(deadline_satisfaction)

	data_point['training_time'] = avg_training_time/len(tasks)
	data_point['sat_ratio'] = avg_sat_ratio/len(tasks)

	return data_point

def avg_across_trials(scheme, state_distribution, experiment_name, trial_indices):
	avg_data = get_empty_data_point()

	for i in trial_indices:
		d = get_data(scheme, state_distribution, experiment_name, i)

		for metric in metrics:
			avg_data[metric] += d[metric]

	for metric in metrics:
		avg_data[metric] = avg_data[metric]/len(trial_indices)

	return avg_data