
dir=$( dirname -- "$0"; )
for method in greedy_search modified_beam_search fast_beam_search; do
  ./$dir/decode.py \
    --epoch 30 \
    --avg 10 \
    --use-averaged-model True \
    --exp-dir ./$dir/exp-B \
    --max-duration 300 \
    --decoding-method $method \
    --max-sym-per-frame 1 \
    --num-encoder-layers 24 \
    --dim-feedforward 1536 \
    --nhead 8 \
    --encoder-dim 384 \
    --decoder-dim 512 \
    --joiner-dim 512
done
