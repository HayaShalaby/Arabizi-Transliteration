#!/bin/bash
setcfg() {
  sed -i "s/^\( *ATTN = \)\"[a-z]*\"/\1\"$1\"/; s/^\( *SCORE = \)\"[a-z]*\"/\1\"$2\"/" seq2seq.py
}
for cfg in "global general" "localp dot" "localp general"; do
  set -- $cfg
  setcfg $1 $2
  grep -E "^ *(ATTN|SCORE) =" seq2seq.py
  for s in 51 53 55; do
    python seq2seq.py $s > log_$1_$2_$s.txt 2>&1
  done
done
setcfg global dot
echo ALL DONE > run_2x2.done
