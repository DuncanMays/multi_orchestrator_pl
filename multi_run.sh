# for T in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
for T in 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60
do
	for W in 4 6 8 10
	do
		echo ------------------------ trial number: $T, num_learners: $W, MED uncertain ------------------------
		python orchestrator.py  -experiment_name high_time_nl_$W -trial_index $T -num_learners $W -data_allocation_regime MED -state_distribution uncertain
		echo ------------------------ trial number: $T, num_learners: $W MMTT uncertain ------------------------
		python orchestrator.py  -experiment_name high_time_nl_$W -trial_index $T -num_learners $W -data_allocation_regime MMTT -state_distribution uncertain
		echo ------------------------ trial number: $T, num_learners: $W MMTT ideal ------------------------
		python orchestrator.py  -experiment_name high_time_nl_$W -trial_index $T -num_learners $W -data_allocation_regime MMTT -state_distribution ideal
	done
done

echo all done!
# echo going to sleep now bye
# systemctl suspend