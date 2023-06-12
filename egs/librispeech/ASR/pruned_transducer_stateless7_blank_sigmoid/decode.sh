dir=$( dirname -- "$0"; )

#for m in greedy_search fast_beam_search modified_beam_search ; do
for m in modified_beam_search ; do
  for epoch in 4; do
    for avg in 2; do
        CUDA_VISIBLE_DEVICES="5" ./$dir/decode.py \
          --epoch $epoch \
          --avg $avg \
          --use-averaged-model 1 \
          --exp-dir ./$dir/exp \
          --blank-sigmoid 1 \
          --feedforward-dims  "1024,1024,2048,2048,1024" \
          --max-duration 300 \
          --decoding-method $m
    done
  done
done