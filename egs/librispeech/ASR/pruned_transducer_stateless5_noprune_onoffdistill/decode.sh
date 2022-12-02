dir=$( dirname -- "$0"; )

for decoding_method in greedy_search modified_beam_search; do
#for decoding_method in modified_beam_search; do
for stream in 1; do
for chunk in 1; do
for left in 128; do
for plambda in 1; do
    ./$dir/decode.py \
            --num-encoder-layers 24 \
            --dim-feedforward 1536 \
            --nhead 8 \
            --encoder-dim 384 \
            --decoder-dim 512 \
            --joiner-dim 512 \
            --simulate-streaming ${stream} \
            --decode-chunk-size ${chunk} \
            --left-context ${left} \
            --causal-convolution 1 \
            --epoch 15 \
            --use-averaged-model True \
            --avg 7 \
            --exp-dir ./$dir/exp-scaled \
            --beam-size 4 \
            --max-sym-per-frame 1 \
            --max-duration 600 \
            --pronouncer-lambda $plambda \
            --decoding-method ${decoding_method}
done
done
done
done
done
