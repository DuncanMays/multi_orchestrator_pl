# load metrics cost, training time, sat. ratio, EOL from files named {SCHEME}_{NUM WORKERS}_workers_n.json where n is the number of trials

from sys import path
path.append('..')
from tasks import tasks

from matplotlib import pyplot as plt

import os
import json

target_dir = './vary_heat'
schemes = ['EOL', 'TT']

def get_data(midstring, index):

	task_names = list(tasks.keys())
	deadlines = {task_name: tasks[task_name]['deadline'] for task_name in task_names}

	# this does not average accross trials, so for more trials we'll need to change this code
	# there is already good code to load the average values from a number of files, which will make this easier

	data = {scheme_name: None for scheme_name in schemes}

	# loads data
	for scheme_name in schemes:
		file_name = scheme_name+'_'+midstring+'_heat_'+index+'.json'
		file_path = os.path.join(target_dir, file_name)

		with open(file_path, 'r') as f:
			data[scheme_name] = json.loads(f.read())

	# we now average accross tasks and global update index

	# these data points hold the averages
	def get_empty_data_point():
		return {
			'training_time': 0,
			'sat_ratio': 0,
			'cost': 0,
			'loss': 0,
			'acc': 0,
		}

	avg_data = {scheme_name: get_empty_data_point() for scheme_name in schemes}

	for scheme_name in schemes:

		# holds data one ach metric for each scheme, averaged across tasks
		scheme_times = {scheme_name: 0 for scheme_name in schemes}
		scheme_sat_ratio = {scheme_name: 0 for scheme_name in schemes}
		scheme_costs = {scheme_name: 0 for scheme_name in schemes}
		scheme_losses = {scheme_name: 0 for scheme_name in schemes}
		scheme_accs = {scheme_name: 0 for scheme_name in schemes}

		for task_name in task_names:

			# list of the time length of each global update cycle in this scheme for this task
			training_times = [dp['time'] for dp in data[scheme_name][task_name]]
			# removing the first data point since it records data prior to any training
			training_times = training_times[1:]

			# averaging accross GUI
			# the number of global update cycles conducted for this task
			num_gui = len(training_times)
			# the average time for this task with this scheme across GUI
			avg_time = sum(training_times)/num_gui

			# we now average accross tasks, weighted with the number of global update cycles
			time_weight = 0
			avg_data[scheme_name]['training_time'] = (num_gui*avg_time + time_weight*scheme_times[scheme_name])/(num_gui + time_weight)
			time_weight = num_gui + time_weight

			# we now calculate the satisfaction ratio
			across_gui = [training_time<tasks[task_name]['deadline'] for training_time in training_times]
			avg_across_gui = sum(across_gui)/num_gui
			# we now average over tasks, weighted with the number of global update cycles
			sr_weight = 0
			avg_data[scheme_name]['sat_ratio'] = (num_gui*avg_across_gui + sr_weight*scheme_sat_ratio[scheme_name])/(num_gui + sr_weight)
			sr_weight = num_gui + sr_weight

			training_costs = [dp['cost'] for dp in data[scheme_name][task_name]]
			avg_cost = sum(training_costs)/num_gui
			cost_weight = 0
			avg_data[scheme_name]['cost'] = (num_gui*avg_cost + cost_weight*scheme_costs[scheme_name])/(num_gui + cost_weight)
			cost_weight = num_gui + cost_weight

			# taking the loss and accuracy of the converged network
			final_loss = data[scheme_name][task_name][-1]['loss']
			loss_weight = 0
			avg_data[scheme_name]['loss'] = (num_gui*final_loss + loss_weight*scheme_losses[scheme_name])/(num_gui + loss_weight)
			loss_weight = num_gui + loss_weight

			# avg_data[scheme_name]['acc']
			final_acc = data[scheme_name][task_name][-1]['acc']
			acc_weight = 0
			avg_data[scheme_name]['acc'] = (num_gui*final_acc + acc_weight*scheme_accs[scheme_name])/(num_gui + acc_weight)
			acc_weight = num_gui + acc_weight

	return avg_data

# metrics are: training_time, cost, sat_ratio, loss, acc

heat_settings = [0.5, 1, 1.5, 2]
index_range = [0, 1, 2]
metric = 'sat_ratio'

line_styles = {'TT': '-', 'EOL': '--'}


Y = {scheme: [] for scheme in schemes}
x = [i for i in range(len(heat_settings))]

for n in heat_settings:

	midstr = str(n)

	data_indices = []
	for i in index_range:
		data_indices.append(get_data(midstr, str(i)))

	metrics = data_indices[0][schemes[0]].keys()
	print(metrics)

	avg_data ={scheme:{metric: 0 for metric in metrics} for scheme in schemes}

	for run in data_indices:
		for scheme in schemes:
			for metric in metrics:
				avg_data[scheme][metric] += run[scheme][metric]/len(index_range)

	print(avg_data)

	for scheme in schemes:
		Y[scheme].append(avg_data[scheme][metric])

for scheme in schemes:
	plt.plot(x, Y[scheme], line_styles[scheme], label=scheme)

plt.title('Satisfaction Ratio versus Worker State Heat', size=30)
plt.ylabel('Satisfaction Ration', size=20)
plt.legend(prop={'size': 20})
plt.ylim((0, 1))
plt.xlabel('Heat', size=20)
plt.xticks(x, heat_settings)

plt.show()