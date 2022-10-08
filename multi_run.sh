for T in 1 2 3 4 5
do
	for H in 0.0001 0.5 1.0 1.5 2.0
	do
		echo -------------------------- trial number: $T, heat param: $H, MED uncertain --------------------------
		python orchestrator.py -data_allocation_regime MED -state_distribution uncertain -heat $H -experiment_name heat_param -trial_index $T
		echo -------------------------- trial number: $T, heat param: $H, MMTT uncertain --------------------------
		python orchestrator.py -data_allocation_regime MMTT -state_distribution uncertain -heat $H -experiment_name heat_param -trial_index $T
		echo -------------------------- trial number: $T, heat param: $H, MMTT ideal --------------------------
		python orchestrator.py -data_allocation_regime MMTT -state_distribution ideal -heat $H -experiment_name heat_param -trial_index $T
	done
done

echo all done!