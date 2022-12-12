dir=$( dirname -- "$0"; )

for decoding_method in greedy_search modified_beam_search; do
#for decoding_method in modified_beam_search; do
for stream in 0 1; do
#for stream in 1; do
for chunk in 1; do
for left in 128; do
for plambda in 0 1; do
#for plambda in 1; do
    ! [[ $decoding_method = "modified_beam_search" && $stream = 1 ]] && \
        [ $plambda = 1 ] && break
    [[ $decoding_method = "modified_beam_search" && $stream = 1 ]] && \
        [ $plambda = 1 ] && CUDA_VISIBLE_DEVICES=""

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
            --epoch 8 \
            --use-averaged-model True \
            --avg 4 \
            --exp-dir ./$dir/exp-scaled-norm \
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
