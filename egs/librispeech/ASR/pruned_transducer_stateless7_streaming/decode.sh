dir=$( dirname -- "$0"; )
#for $m in greedy_search fast_beam_search modified_beam_search; do
for m in greedy_search; do
  ./$dir/decode.py \
    --epoch 30 \
    --avg 9 \
    --use-averaged-model 1 \
    --exp-dir ./$dir/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method $m
done