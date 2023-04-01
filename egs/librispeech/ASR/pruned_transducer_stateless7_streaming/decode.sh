dir=$( dirname -- "$0"; )
#for $m in greedy_search fast_beam_search modified_beam_search; do
for m in greedy_search; do
  ./$dir/decode.py \
    --epoch 1 \
    --avg 1 \
    --use-averaged-model 0 \
    --exp-dir ./$dir/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method $m
done