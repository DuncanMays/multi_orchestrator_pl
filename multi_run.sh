# for T in 12
# do
# 	for W in 9
# 	do
# 		echo ------------------------------- trial number $T, $W workers MED uncertain -------------------------------
# 		python orchestrator.py -data_allocation_regime MED -ideal_worker_state False -num_learners $W -experiment_name workers -trial_index $T
# 		echo ------------------------------- trial number $T, $W workers TT uncertain -------------------------------
# 		python orchestrator.py -data_allocation_regime TT -ideal_worker_state False -num_learners $W -experiment_name workers -trial_index $T
# 		echo ------------------------------- trial number $T, $W workers TT ideal -------------------------------
# 		python orchestrator.py -data_allocation_regime TT -ideal_worker_state True -num_learners $W -experiment_name workers -trial_index $T
# 		echo ------------------------------- trial number $T, $W workers MMTT uncertain -------------------------------
# 		python orchestrator.py -data_allocation_regime MMTT -ideal_worker_state False -num_learners $W -experiment_name workers -trial_index $T
# 		echo ------------------------------- trial number $T, $W workers MMTT ideal -------------------------------
# 		python orchestrator.py -data_allocation_regime MMTT -ideal_worker_state True -num_learners $W -experiment_name workers -trial_index $T
# 	done
# done

for T in 26
do
	for W in 3 5 7 9
	do
		echo ------------------------------- trial number $T, $W workers MED uncertain -------------------------------
		python orchestrator.py -data_allocation_regime MED -ideal_worker_state False -num_learners $W -experiment_name workers -trial_index $T
		echo ------------------------------- trial number $T, $W workers TT uncertain -------------------------------
		python orchestrator.py -data_allocation_regime TT -ideal_worker_state False -num_learners $W -experiment_name workers -trial_index $T
		echo ------------------------------- trial number $T, $W workers TT ideal -------------------------------
		python orchestrator.py -data_allocation_regime TT -ideal_worker_state True -num_learners $W -experiment_name workers -trial_index $T
		echo ------------------------------- trial number $T, $W workers MMTT uncertain -------------------------------
		python orchestrator.py -data_allocation_regime MMTT -ideal_worker_state False -num_learners $W -experiment_name workers -trial_index $T
		echo ------------------------------- trial number $T, $W workers MMTT ideal -------------------------------
		python orchestrator.py -data_allocation_regime MMTT -ideal_worker_state True -num_learners $W -experiment_name workers -trial_index $T
	done
done

echo all done!