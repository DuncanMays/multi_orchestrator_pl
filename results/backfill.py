import os
import sys

from itertools import product

target_dir = 'Nov_28_week'

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
	# trial_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
	trial_indices = range(1, 31)
	# num_workers = [3, 5, 7, 9]
	num_workers = [5, 7, 9, 11, 13]
	scheme_state_prefix = ['MED_uncertain', 'MMTT_uncertain', 'MMTT_ideal', 'MMTT_static']
	experiment_name = 'num_learners'

	for (prefix, nw, t) in product(scheme_state_prefix, num_workers, trial_indices):

		file_name = prefix + "_" + str(nw) + "_" + experiment_name + "_" + str(t) + '.json'
		file_path = os.path.join(target_dir, file_name)

		if not test_file(file_path):
			# if the file does not exist, we need to assemble a command to run orchestrator to run the corresponding trial
			# for example:
			# python orchestrator.py -data_allocation_regime EOL -ideal_worker_state False -num_learners $W -experiment_name workers -trial_index $T

			dar, wrkr_dst = prefix.split('_')

			command = 'python orchestrator.py -data_allocation_regime '+dar+' -state_distribution '+wrkr_dst+' -num_learners '+str(nw)+' -trial_index '+str(t)+' -experiment_name '+experiment_name
			command_list.append(command)

	return command_list

if (__name__ == '__main__'):
	cl = get_command_list()

	for c in cl:
		print(c)

	# if ('-r' in sys.argv):
	# 	print('running commands')