from matplotlib import pyplot as plt
import json
import os

# the directory we'll be loading data from
src_dir = 'EOL_line_plot_0'
# the filenames in the source directory
file_names = os.listdir(src_dir)
# we only want the ones that end in .json
file_names = list(filter(lambda name : name.split('.')[-1] == 'json', file_names))
# appending src_dir onto the filenames
file_names = [os.path.join(src_dir, name) for name in file_names]

# this dict holds the average value for all data points accross the training runs recorded in src_dir
averages = {}
num_training_runs = len(file_names)

with open(file_names.pop(), 'r') as f:
	averages = json.loads(f.read())

for file_name in file_names:
	with open(file_name, 'r') as f:
		# new data is a dict of lists, each list represents a training run for a certain task
		new_data = json.loads(f.read())

		# iterating over tasks
		for task_name in new_data:
			# iterating over data points for that task
			for i in range(len(new_data[task_name])):
				# we're gonna do each of these separately so that it's easy to opt out for things we don't want to average
				averages[task_name][i]['time'] += new_data[task_name][i]['time']
				averages[task_name][i]['acc'] += new_data[task_name][i]['acc']
				averages[task_name][i]['loss'] += new_data[task_name][i]['loss']

# normalizing
for task_name in averages:
	for i in range(len(averages[task_name])):
		averages[task_name][i]['time'] = averages[task_name][i]['time']/num_training_runs
		averages[task_name][i]['acc'] = averages[task_name][i]['acc']/num_training_runs
		averages[task_name][i]['loss'] = averages[task_name][i]['loss']/num_training_runs

# the global update indices of each data point, is the same for all tasks
gui_axis = [i for i in range(len(list(averages.values())[0]))]

# the time for each global update, is different for different tasks
time_axes = {}
for task_name in averages:
	# update times are recorded as the interval since the last update completed, we want the total time since the beginning of training
	total_time = 0
	# the array that holds the time each global update completed, starts at zero
	t = []

	for dp in averages[task_name]:
		total_time += dp['time']
		t.append(total_time)

	time_axes[task_name] = t

for task_name in averages:
	plt.plot(gui_axis, [dp['acc'] for dp in averages[task_name]], label=task_name)

plt.legend()
plt.title('EOL')
plt.ylabel('Accuracy (fraction)')
plt.xlabel('Global Update Index')

plt.show()

exit()

# we now plot the results
fig, axs = plt.subplots(2, 2)

for task_name in averages:
	axs[0][0].plot(gui_axis, [dp['loss'] for dp in averages[task_name]], label=task_name+'_loss')

axs[0][0].legend()
axs[0][0].set_xlabel('global update index')
axs[0][0].set_ylabel('loss')

for task_name in averages:
	axs[1][0].plot(gui_axis, [dp['acc'] for dp in averages[task_name]], label=task_name+'_acc')

axs[1][0].legend()
axs[1][0].set_xlabel('global update index')
axs[1][0].set_ylabel('accuracy')

for task_name in averages:
	axs[0][1].plot(time_axes[task_name], [dp['loss'] for dp in averages[task_name]], label=task_name+'_loss')

axs[0][1].legend()
axs[0][1].set_xlabel('time (s)')
axs[0][1].set_ylabel('loss')

for task_name in averages:
	axs[1][1].plot(time_axes[task_name], [dp['acc'] for dp in averages[task_name]], label=task_name+'_acc')

axs[1][1].legend()
axs[1][1].set_xlabel('time (s)')
axs[1][1].set_ylabel('accuracy')

plt.show()