dir=$( dirname -- "$0"; )

#for m in greedy_search; do
for m in greedy_search; do
  for epoch in 70; do
    for avg in 1; do
          CUDA_VISIBLE_DEVICES="2" ./$dir/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --causal 1 \
            --num-workers 8 \
            --chunk-size 8 \
            --left-context-frames 128 \
            --exp-dir $dir/exp-small-from50-noeval-d-nl2-ls0.01-decrawrnn-flow-dep4-dim256 \
            --rnnt-type "regular" \
            --num-encoder-layers 2,2,2,2,2,2 \
            --feedforward-dim 512,768,768,768,768,768 \
            --encoder-dim 192,256,256,256,256,256 \
            --encoder-unmasked-dim 192,192,192,192,192,192 \
            --train-in-eval-mode 0 \
            --use-encoder-pred 1 \
            --encoder-pred-detach 1 \
            --encoder-pred-bottleneck-dim 256 \
            --encoder-pred-kernel-size 17 \
            --encoder-pred-num-layers 2 \
            --encoder-pred-loss-scale 0.01 \
            --encoder-pred-logp-scale 0.0 \
            --encoder-pred-logp-ratio-clamp 0.0 \
            --encoder-pred-enc-in-rnn 0 \
            --encoder-pred-dec-in-rnn 1 \
            --encoder-pred-dec-in-raw 1 \
            --encoder-pred-flow-depth 4 \
            --encoder-pred-flow-num-blocks 2 \
            --encoder-pred-flow-hidden-dim 256 \
            --max-duration 600 \
            --decoding-method $m 
    done
  done
done