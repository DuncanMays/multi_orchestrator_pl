import json
import sys
import os
import numpy as np

sys.path.append('..')
from tasks import tasks as tasks

default_target_dir = 'Nov_28_week'

metrics = ['time_prediction_error', 'time_prediction', 'max_grad_div', 'mean_grad_div', 'resource_util', 'max_training_time', 'loss', 'acc', 'cost', 'total_training_time', 'training_time', 'worker_sat_ratio', 'task_sat_ratio']
task_names = list(tasks.keys())

def get_empty_data_point():
	return {metric: 0 for metric in metrics}

def get_acc_data(scheme, state_distribution, experiment_name, trial_number, task_name, target_dir=default_target_dir):
	filename = scheme+'_'+state_distribution+'_'+experiment_name+'_'+str(trial_number)+'.json'
	filepath = os.path.join('.', target_dir, filename)

	with open(filepath, 'r') as f:
		raw_data = json.loads(f.read())

		return [d['acc'] for d in raw_data['training_metrics'][task_name]]

MMTT = 0
mmtt = 0
MED = 0
med = 0

# Given a scheme (MED vs TT), a state distribution (ideal/uncertain), experiment name and trial number, report the cost, EOL, final loss and acc, as well as the average training time and sat ratio
# each call to this function reads data from one training run
def get_data(scheme, state_distribution, experiment_name, trial_number, target_dir=default_target_dir):
	global MED, MMTT, med, mmtt

	filename = scheme+'_'+state_distribution+'_'+experiment_name+'_'+str(trial_number)+'.json'
	filepath = os.path.join('.', target_dir, filename)

	with open(filepath, 'r') as f:
		raw_data = json.loads(f.read())

	task_names = raw_data['training_metrics'].keys()

	data_point = get_empty_data_point()

	# EOL and cost are easy to obtain
	data_point['cost'] = raw_data['cost']

	# averaging converged loss and acc accross tasks
	data_point['loss'] = sum([raw_data['training_metrics'][task_name][-2]['loss'] for task_name in task_names])/len(task_names)
	data_point['acc'] = sum([raw_data['training_metrics'][task_name][-2]['acc'] for task_name in task_names])/len(task_names)

	avg_training_time = 0
	avg_max_training_time = -1
	avg_task_sat_ratio = 0
	avg_worker_sat_ratio = 0
	avg_total_training_time = 0
	avg_learner_util = 0
	max_grad_div = 0
	mean_grad_div = 0

	for task_name in task_names:
		# ditching the first index as that has training time zero
		# training times is a list of lists, the first dimension moves accross global update cycles, the second dimension accross learners
		training_times = [d['times'] for d in raw_data['training_metrics'][task_name][1:]]

		# the average training time for each worker
		avg_times = [sum(t)/len(t) for t in training_times]
		avg_training_time += sum(avg_times)/len(avg_times)

		# the max training time of each worker in a global update, this is the time for the global update
		max_times = [max(t) for t in training_times]
		avg_max_training_time += sum(max_times)/len(max_times)

		# this metric gives the total amount of time until convergence
		# calculated as the sum of the amount of time for the longest running learner in each global update cycle
		total_training_time = sum([max(t) for t in training_times])
		avg_total_training_time += total_training_time

		# the satisfaction ration is the number of workers who returned before the deadline
		# deadline_offset = float(experiment_name.split('_')[0])
		satisfied_deadlines = sum([sum([t < tasks[task_name]['deadline'] for t in T]) for T in training_times])
		total_deadlines = sum([len(t) for t in training_times])
		avg_worker_sat_ratio += satisfied_deadlines/total_deadlines

		# adds a 1 or 0 depending if the task finished before the deadline this trial
		if (total_training_time < len(training_times)*tasks[task_name]['deadline']):
			avg_task_sat_ratio += 1

		# we now calculate the resource utilization using the max time of each GU cycle in the training run
		learner_utils = [[t/max_times[i] for t in T] for i, T in enumerate(training_times)]
		learner_utils = [sum(t)/len(t) for t in learner_utils]
		avg_learner_util += sum(learner_utils)/len(learner_utils)

		# the average max and mean gradient divergence for each GU in raw_data
		max_grad_div += sum([d['max_div'] for d in raw_data['training_metrics'][task_name]])/len(training_times)
		mean_grad_div += sum([d['mean_div'] for d in raw_data['training_metrics'][task_name]])/len(training_times)

		avg_error = 0
		for t in training_times:
			predictions = raw_data['time_prediction'][task_name]
			avg_error = sum(abs(predictions[i] - t[i])/t[i] for i in range(len(t)))/len(t)

		data_point['time_prediction_error'] += avg_error
		
	data_point['training_time'] = avg_training_time/len(task_names)
	data_point['max_training_time'] = avg_max_training_time/len(task_names)
	data_point['total_training_time'] = avg_total_training_time/len(task_names)
	data_point['worker_sat_ratio'] = avg_worker_sat_ratio/len(task_names)
	data_point['task_sat_ratio'] = avg_task_sat_ratio/len(task_names)
	data_point['resource_util'] = avg_learner_util/len(task_names)
	data_point['max_grad_div'] = max_grad_div/len(task_names)
	data_point['mean_grad_div'] = mean_grad_div/len(task_names)
	data_point['time_prediction_error'] = data_point['time_prediction_error']/len(task_names)

	return data_point

def get_mean(l):
	return sum(l)/len(l)

def get_stand_dev(l):
	return np.sqrt(np.var(l))

def avg_across_trials(scheme, state_distribution, experiment_name, trial_indices, target_dir=default_target_dir):
	avg_data = get_empty_data_point()
	data_list = []

	for i in trial_indices:
		d = get_data(scheme, state_distribution, experiment_name, i, target_dir)
		data_list.append(d)

	# data_list is a list accross trials of dicts with attributtes for each metric
	# data_dict is a dict with attributtes for each metric of lists accross trials
	data_dict = {metric: [d[metric] for d in data_list] for metric in metrics}

	data_mean = {metric: get_mean(data_dict[metric]) for metric in metrics}
	data_stand_dev = {metric: get_stand_dev(data_dict[metric]) for metric in metrics}

	for metric in metrics:
		avg_data[metric] = data_mean[metric]
		avg_data[metric+'_stand_dev'] = data_stand_dev[metric]

	return avg_data