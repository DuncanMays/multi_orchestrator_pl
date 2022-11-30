
for T in 1 2 3 4 5 6 7 8 9 10
do
	echo ------------------------ trial number: $T, MED uncertain ------------------------
	python orchestrator.py -data_allocation_regime MED -state_distribution uncertain -experiment_name 201510_deadline -trial_index $T
	echo ------------------------ trial number: $T, MMTT uncertain ------------------------
	python orchestrator.py -data_allocation_regime MMTT -state_distribution uncertain -experiment_name 201510_deadline -trial_index $T
	echo ------------------------ trial number: $T, MMTT ideal ------------------------
	python orchestrator.py -data_allocation_regime MMTT -state_distribution ideal -experiment_name 201510_deadline -trial_index $T
done

echo all done!
# echo going to sleep now bye
# systemctl suspend