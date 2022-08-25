for T in 6 7 8 9 10
do
	echo ------------------------------------ trial number $T EOL uncertain ------------------------------------
	python orchestrator.py -data_allocation_regime EOL -ideal_worker_state False -experiment_name 9_workers -trial_index $T
	echo ------------------------------------ trial number $T EOL_max uncertain ------------------------------------
	python orchestrator.py -data_allocation_regime EOL_max -ideal_worker_state False -experiment_name 9_workers -trial_index $T
	echo ------------------------------------ trial number $T TT uncertain ------------------------------------
	python orchestrator.py -data_allocation_regime TT -ideal_worker_state False -experiment_name 9_workers -trial_index $T
	echo ------------------------------------ trial number $T TT ideal ------------------------------------
	python orchestrator.py -data_allocation_regime TT -ideal_worker_state True -experiment_name 9_workers -trial_index $T
done

echo all done!