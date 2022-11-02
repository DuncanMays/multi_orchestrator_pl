for T in 19
do
	for d in 40 50
	do
		echo ------------------------ trial number: $T, deadline_adjust $d, MED uncertain ------------------------
		python orchestrator.py -data_allocation_regime MED -state_distribution uncertain -deadline_adjust $d -experiment_name deadline -trial_index $T
		echo ------------------------ trial number: $T, deadline_adjust $d, MMTT uncertain ------------------------
		python orchestrator.py -data_allocation_regime MMTT -state_distribution uncertain -deadline_adjust $d -experiment_name deadline -trial_index $T
		echo ------------------------ trial number: $T, deadline_adjust $d, MMTT ideal ------------------------
		python orchestrator.py -data_allocation_regime MMTT -state_distribution ideal -deadline_adjust $d -experiment_name deadline -trial_index $T
	done
done

for T in 20 21 22 23 24 25 26 27 28 29 20 21 22 23 24 25 26 27 28 29 30
do
	for d in 0 10 20 30 40 50
	do
		echo ------------------------ trial number: $T, deadline_adjust $d, MED uncertain ------------------------
		python orchestrator.py -data_allocation_regime MED -state_distribution uncertain -deadline_adjust $d -experiment_name deadline -trial_index $T
		echo ------------------------ trial number: $T, deadline_adjust $d, MMTT uncertain ------------------------
		python orchestrator.py -data_allocation_regime MMTT -state_distribution uncertain -deadline_adjust $d -experiment_name deadline -trial_index $T
		echo ------------------------ trial number: $T, deadline_adjust $d, MMTT ideal ------------------------
		python orchestrator.py -data_allocation_regime MMTT -state_distribution ideal -deadline_adjust $d -experiment_name deadline -trial_index $T
	done
done

echo all done!
# echo going to sleep now bye
# systemctl suspend