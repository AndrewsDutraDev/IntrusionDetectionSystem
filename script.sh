#!/bin/sh


# cd 'with-attack']
cd 'without-attack'
ls

for f in 'monday' 'tuesday' 'wednesday' 'thursday' 'friday'
do
	cd $f
	echo "Processing $f"
	ls

	echo '--------------------------------'

	tcpstat -r inside.tcpdump 300 -o "Time:%S\tn=%n\tavg=%a\tstddev=%d\tbps=%b\n" >> inside.txt

	tcpstat -r outside.tcpdump 300 -o "Time:%S\tn=%n\tavg=%a\tstddev=%d\tbps=%b\n" >> outside.txt

	cat inside.txt outside.txt > "$f"_in_out.txt

	mv "$f"_in_out.txt ../

done

cd ../

cat monday_in_out.txt tuesday_in_out.txt wednesday_in_out.txt thursday_in_out.txt friday_in_out.txt > all_days_in_out.txt
