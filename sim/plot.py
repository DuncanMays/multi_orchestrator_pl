from matplotlib import pyplot as plt
from types import SimpleNamespace

import sys
import os
import json

arg_list = sys.argv[1:]
arg_dict = {}
for index, arg in enumerate(arg_list):
	if (arg[0] == '-'):
		arg_dict[arg[1:]] = arg_list[index+1]

args = SimpleNamespace(**arg_dict)

results_folder = './Sep_29_meeting/mnist_cnn_data'

def get_average_acc(w, sr):
	y = []

	filename = str(w)+'_workers_'+str(sr)+'_sat_ratio.json'
	filepath = os.path.join(results_folder, filename)

	accs = None
	with open(filepath, 'r') as f:
		# the accuracies at every global update cycle for each trial
		all_accs = json.loads(f.read())
		# the final accuracy at every global update cycle for each trial
		accs = [a[-1] for a in all_accs]

	print(len(accs))

	avg_acc = sum(accs)/len(accs)

	return avg_acc

W = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sat_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

x = [i+1 for i in range(len(W))]

for sr in sat_ratios:
	y = []

	for w in W:
		y.append(get_average_acc(w, sr))
	
	plt.plot(x, y, label='sat_ratio: '+str(sr))

plt.title('Final Accuracy Over Varying Number of Workers', size=20)
plt.xlabel('num workers', size=15)
plt.ylabel('accuracy (fraction)', size=15)
plt.legend(prop={'size': 15})
plt.show()