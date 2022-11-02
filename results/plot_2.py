from read_data import avg_across_trials
from matplotlib import pyplot as plt

trial_indices = list(range(1, 30))

# metrics = ['time_prediction', 'time_prediction_error', 'max_grad_div', 'mean_grad_div', 'resource_util', 'max_training_time', 'training_time', 'acc', 'loss', 'cost', 'worker_sat_ratio', 'task_sat_ratio', 'total_training_time']
metrics = ['time_prediction_error', 'max_grad_div', 'resource_util', 'max_training_time', 'worker_sat_ratio']
metric_titles = {'time_prediction_error': 'time_prediction_error', 'max_grad_div': 'Parameter Divergence', 'resource_util': 'Occupancy', 'max_training_time': 'Training Time', 'worker_sat_ratio': 'Satisfaction Ratio'}
scheme_state_dists = [('MMTT', 'ideal'), ('MED', 'uncertain'), ('MMTT', 'uncertain')]
experiment_details = [('Oct_28_meeting/State_Heat_1.5', 'state_heat'), ('Oct_28_meeting/stop_halting', 'stop_halting')]

def get_experiment_data(folder, experiment_name):
	averaged_results = {}

	for metric in metrics:
		metric_results = []
		for (scheme, state_dist) in scheme_state_dists:
			# scheme, state_distribution, experiment_name, trial_number, target_dir
			y = avg_across_trials(scheme, state_dist, experiment_name, trial_indices, folder)[metric]
			metric_results.append(y)

		averaged_results[metric] = metric_results

	return averaged_results

def plot_all_data(data, title):
	x = [i for i in range(len(metrics))]
	y = []

	for metric in metrics:
		metric_data = data[metric]

		y.append(metric_data)

	y_T = [[y[i][j] for i in range(len(metrics))] for j in range(3)]

	# for i in range(3):
	# 	print(y_T)

	offset = -0.2
	step = 0.2
	for i in range(3):
		plt.bar([a + offset for a in x], y_T[i], width=0.2)
		offset += step

	plt.title(title)
	plt.xticks(x, [m[0:4] for m in metrics])

def main():

	for folder, name in [experiment_details[1]]:
		print(folder, name)
		data = get_experiment_data(folder, name)

		for metric in metrics:
			metric_data = data[metric]
			x = list(range(len(metric_data)))
			plt.bar(x, metric_data, color=['blue', 'orange', 'green'])

			plt.title(metric_titles[metric], size=15)
			plt.xticks(x, ['MMTT_ideal', 'MED', 'MMTT_uncertain'], size=15)

			if (metric == 'worker_sat_ratio'):
				plt.ylim(0.8, 1.0)

			if (metric == 'resource_util'):
				plt.ylim(0.6, 1.0)

			plt.savefig(metric+'_'+name+'.png')
			# plt.show()
			plt.clf()

if (__name__ == '__main__'):
	main()