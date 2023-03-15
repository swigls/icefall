dir=$( dirname -- "$0"; )

for m in greedy_search; do
  for epoch in 30; do
    for avg in 9; do
      ./$dir/decode.py \
          --epoch $epoch \
          --avg $avg \
          --use-averaged-model 1 \
          --exp-dir ./$dir/exp \
          --feedforward-dims  "1024,1024,2048,2048,1024" \
          --max-duration 480 \
          --max-sym-per-frame 1000 \
          --decoding-method $m
    done
  done
done
