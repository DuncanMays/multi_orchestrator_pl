from plot import get_averages
from matplotlib import pyplot as plt

import os
OF = 'RSS'

target_dir = './'+OF+'_vary_workers_2_4'
target_files = os.listdir(target_dir)
# we only want the ones that end in .json
target_files = list(filter(lambda name : name.split('.')[-1] == 'json', target_files))
target_files = [os.path.join(target_dir, name) for name in target_files]

# we now sort the filenames into groups depending on how many workers were used in each training run
worker_populations = [2, 3, 4]

sorted_files = []
for _ in worker_populations:
	sorted_files.append([])

for name in target_files:
	num_workers = name.split('/')[-1][0]
	index = worker_populations.index(int(num_workers))
	sorted_files[index].append(name)

# we can now rely on sorted_files being in order of worker population
averages = []
for file_list in sorted_files:
	averages.append(get_averages(file_list))

task_names = averages[0].keys()
# we now split the data based on task, accross worker populations
data_per_task = {task_name: [] for task_name in task_names}
num_tasks = len(task_names)

# each task will have an offset so the bars for multiple tasks don't overlap
task_offsets = {}
offset_interval = 0.9/num_tasks
offset = -0.225
for task_name in task_names:
	task_offsets[task_name] = offset
	offset += offset_interval

# the training deadlines at the time the data was recorded
training_deadlines = {'mnist_ffn':30, 'mnist_cnn':30}

for data_dict in averages:
	for task_name in task_names:
		# data_dict[task_name] is a list of the accuracy, time and loss at each global training cycle accross the training run
		# we must average the training times accross each training run, excluding the first one

		# plot the training cycle time versus num workers
		# training_times = [datum['time'] for datum in data_dict[task_name][1:]]
		# data_point = sum(training_times)/len(training_times)

		# plot the satisfaction ratio versus num workers
		training_times = [datum['time'] for datum in data_dict[task_name][1:]]
		num_satisfied = 0
		for t in training_times:
			if t <= training_deadlines[task_name]:
				num_satisfied += 1

		data_point = num_satisfied/len(training_times)

		data_per_task[task_name].append(data_point)

for task_name in task_names:
	x = [p + task_offsets[task_name] for p in worker_populations]
	plt.bar(x, data_per_task[task_name], label=task_name, width=0.8/num_tasks)

plt.title(OF)
plt.ylabel('Satisfaction Ratio')
plt.xlabel('number of workers')
plt.xticks(worker_populations)
plt.legend()
plt.show()
