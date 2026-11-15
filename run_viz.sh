#!/bin/bash
while [ ! -f run_multip.done ]; do sleep 30; done
for a in global localp; do
  sed -i "s/^\( *ATTN = \)\"[a-z]*\"/\1\"$a\"/" seq2seq_viz.py
  python seq2seq_viz.py 51 > viz_log_$a.txt 2>&1
done
echo ALL DONE > run_viz.done
