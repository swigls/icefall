dir=$( dirname -- "$0"; )

decoding_method="greedy_search"  # "fast_beam_search", "modified_beam_search"
#decoding_method="modified_search"  
for chunk in 1; do
  for left in 128; do
    ./$dir/decode.py \
            --num-encoder-layers 24 \
            --dim-feedforward 1536 \
            --nhead 8 \
            --encoder-dim 384 \
            --decoder-dim 512 \
            --joiner-dim 512 \
            --simulate-streaming 1 \
            --decode-chunk-size ${chunk} \
            --left-context ${left} \
            --causal-convolution 1 \
            --epoch 2 \
            --use-averaged-model True \
            --avg 1 \
            --exp-dir ./$dir/exp-B-uni \
            --max-sym-per-frame 1 \
            --max-duration 600 \
            --decoding-method ${decoding_method}
  done
done
