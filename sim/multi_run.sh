
for sat_ratio in 1.0 0.8 0.6 0.4 0.2
do
	for W in 1 2 3 4 5 6 7 8 9 10
	do
		echo --------------------- $W workers, $sat_ratio sat_ratio ---------------------
		python sim.py -f ${W}_workers_${sat_ratio}_sat_ratio.json -k $W -t 15 -sr $sat_ratio
	done
done

echo all done!