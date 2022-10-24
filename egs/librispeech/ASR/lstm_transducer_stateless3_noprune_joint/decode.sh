dir=$( dirname -- "$0"; )

for decoding_method in greedy_search fast_beam_search modified_beam_search; do
  ./$dir/decode.py \
    --epoch 2 \
    --avg 1 \
    --exp-dir $dir/exp \
    --max-duration 300 \
    --num-encoder-layers 12 \
    --rnn-hidden-size 1024 \
    --decoding-method $decoding_method \
    --use-averaged-model True \
    --beam 4 \
    --max-contexts 4 \
    --max-states 8 \
    --beam-size 4
done
