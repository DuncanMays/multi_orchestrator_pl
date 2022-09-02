for T in 16 17 18 19 20
do
	for W in 3 5 7 9
	do
		echo ------------------------------- trial number $T $W workers EOL uncertain -------------------------------
		python orchestrator.py -data_allocation_regime EOL -ideal_worker_state False -num_learners $W -experiment_name workers -trial_index $T
		echo ------------------------------- trial number $T $W workers EOL_max uncertain -------------------------------
		python orchestrator.py -data_allocation_regime EOL_max -ideal_worker_state False -num_learners $W -experiment_name workers -trial_index $T
		echo ------------------------------- trial number $T $W workers TT uncertain -------------------------------
		python orchestrator.py -data_allocation_regime TT -ideal_worker_state False -num_learners $W -experiment_name workers -trial_index $T
		echo ------------------------------- trial number $T $W workers TT ideal -------------------------------
		python orchestrator.py -data_allocation_regime TT -ideal_worker_state True -num_learners $W -experiment_name workers -trial_index $T
	done
done

echo all done!