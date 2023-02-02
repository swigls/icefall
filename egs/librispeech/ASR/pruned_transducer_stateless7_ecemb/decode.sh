dir=$( dirname -- "$0"; )

#for m in greedy_search modified_beam_search; do
for m in greedy_search; do
  for epoch in 28; do
    for avg in 9; do
      ./$dir/decode.py \
          --epoch $epoch \
          --avg $avg \
          --use-averaged-model 1 \
          --manifest-dir data/ecemb_24.0 \
          --exp-dir ./$dir/exp-ecemb_24.0 \
          --feedforward-dims  "1024,1024,2048,2048,1024" \
          --max-duration 600 \
          --max-sym-per-frame 1 \
          --decoding-method $m
    done
  done
done
