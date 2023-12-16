
import os
import sys

from itertools import product

target_dir = 'vary_tasks'

# return a boolean indicating if a file exists at the given path
def test_file(filepath):

	try:
		# tests if the file exists and can be read
		with open(filepath, 'r') as f:
			f.read()

		return True

	except(FileNotFoundError):
		return False

def get_command_list():

	command_list = []
	trial_indices = range(1, 41)
	num_tasks = [1, 2, 3, 4]
	# num_workers = [4, 6, 8, 10]
	scheme_state_prefix = ['MED_uncertain', 'MMTT_uncertain', 'MMTT_ideal']
	experiment_name = 'third_vary_tasks'

	# for (prefix, nw, t) in product(scheme_state_prefix, num_workers, trial_indices):
	for (prefix, nw, t) in product(scheme_state_prefix, num_tasks, trial_indices):

		file_name = prefix + "_" + experiment_name + "_" + str(nw) + "_" + str(t) + '.json'
		file_path = os.path.join(target_dir, file_name)

		if not test_file(file_path):
			# if the file does not exist, we need to assemble a command to run orchestrator to run the corresponding trial
			# for example:
			# python orchestrator.py -data_allocation_regime EOL -ideal_worker_state False -num_learners $W -experiment_name workers -trial_index $T

			dar, wrkr_dst = prefix.split('_')

			command = 'python orchestrator.py -data_allocation_regime '+dar+' -state_distribution '+wrkr_dst+' -num_tasks '+str(nw)+' -trial_index '+str(t)+' -experiment_name '+experiment_name + "_" + str(nw) + ' -new_states True'
			command_list.append(command)

	return command_list

if (__name__ == '__main__'):
	cl = get_command_list()

	for c in cl:
		print(c)

	print('missed trials:', len(cl))

	# if ('-r' in sys.argv):
	# 	print('running commands')