#!/bin/bash
for s in 51 53 55; do
  /usr/bin/time -f "seed $s: %e s" python seq2seq.py $s > log_multip_dot_$s.txt 2> time_multip_$s.txt
done
echo ALL DONE > run_multip.done
