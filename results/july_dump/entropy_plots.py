from plot import get_averages
from matplotlib import pyplot as plt

import os

target_dir = './state_entropy'
target_files = os.listdir(target_dir)
# we only want the ones that end in .json
target_files = list(filter(lambda name : name.split('.')[-1] == 'json', target_files))
target_files = [os.path.join(target_dir, name) for name in target_files]

# we now sort the filenames into groups depending on the entropy of the worker state distribution
high_entropy = []
low_entropy = []

for path in target_files:
	entropy = path.split('/')[-1].split('_')[3]
	
	if (entropy == 'low'):
		low_entropy.append(path)

	elif (entropy == 'high'):
		high_entropy.append(path)

# we now sort filenames depending on the data allocation method
DAMs = ['EOL', 'RSS', 'MMET']
def sort_by_DAM(file_paths):
	sorted_paths = [[] for _ in DAMs]

	for path in file_paths:
		# data allocation method
		DAM = path.split('/')[-1].split('_')[0]

		index = DAMs.index(DAM)
		sorted_paths[index].append(path)

	return sorted_paths

low_entropy_sorted = sort_by_DAM(low_entropy)
high_entropy_sorted = sort_by_DAM(high_entropy)

low_entropy_data = [get_averages(files) for files in low_entropy_sorted]
high_entropy_data = [get_averages(files) for files in high_entropy_sorted]

task_names = high_entropy_data[0].keys()
num_tasks = len(task_names)
data_per_task = {task_name: [] for task_name in task_names}

for data_dict in high_entropy_data:
	for task_name in task_names:
		# data_dict[task_name] is a list of the accuracy, time and loss at each global training cycle accross the training run
		# we must average the training times accross each training run, excluding the first one

		# plot the training cycle time versus num workers
		training_times = [datum['time'] for datum in data_dict[task_name][1:]]
		data_point = sum(training_times)/len(training_times)

		# # plot the satisfaction ratio versus num workers
		# training_times = [datum['time'] for datum in data_dict[task_name][1:]]
		# num_satisfied = 0
		# for t in training_times:
		# 	if t <= training_deadlines[task_name]:
		# 		num_satisfied += 1

		# data_point = num_satisfied/len(training_times)

		data_per_task[task_name].append(data_point)

x = [i for i in range(len(DAMs))]
offset = -0.25
for task_name in task_names:
	x_offset = [i + offset for i in x]
	offset += 1/num_tasks
	plt.bar(x_offset, data_per_task[task_name], label=task_name, width=0.8/num_tasks)

plt.title('High Entropy State Distribution')
plt.ylabel('Training Cycle Time (s)')
plt.xlabel('Data Allocation Method')
plt.xticks(x, DAMs)
plt.legend()
plt.show()

# plot the training cycle time versus state entropy

# plot the satisfaction ratio versus state entropy