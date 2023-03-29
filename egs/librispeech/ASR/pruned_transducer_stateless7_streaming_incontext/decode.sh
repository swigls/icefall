dir=$( dirname -- "$0"; )

for m in greedy_search; do
  for epoch in 15; do
    for avg in 4; do
      ./$dir/decode.py \
        --epoch $epoch \
        --avg $avg \
        --exp-dir ./$dir/exp \
        --max-duration 600 \
        --max-sym-per-frame 300 \
        --decode-chunk-len 8000 \
        --decoding-method $m
    done
  done
done
