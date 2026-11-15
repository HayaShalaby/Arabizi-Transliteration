#!/bin/bash
while pgrep -f "python seq2seq.py" > /dev/null; do sleep 20; done
for s in 51 53 55; do ATTN=localm SCORE=dot python seq2seq.py $s > log_localm_$s.txt 2>&1 & done
wait
echo ALL DONE > run_localm.done
