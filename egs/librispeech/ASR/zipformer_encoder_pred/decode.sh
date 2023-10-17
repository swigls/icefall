dir=$( dirname -- "$0"; )

for m in greedy_search; do
  for epoch in 30; do
    for avg in 1; do
      for logp_scale in 0; do
          CUDA_VISIBLE_DEVICES="3" ./$dir/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --causal 1 \
            --num-workers 8 \
            --chunk-size 8 \
            --left-context-frames 256 \
            --exp-dir $dir/exp-small-noev-encraw-d-nl8-dim128-noise0.1-ls1.0-decrawrnn-flow-dep2-dim320-from20-lps0.05 \
            --rnnt-type "regular" \
            --num-encoder-layers 2,2,2,2,2,2 \
            --feedforward-dim 512,768,768,768,768,768 \
            --encoder-dim 192,256,256,256,256,256 \
            --encoder-unmasked-dim 192,192,192,192,192,192 \
            --train-in-eval-mode 1 \
            --use-encoder-pred 1 \
            --encoder-pred-detach 1 \
            --encoder-pred-bottleneck-dim 128 \
            --encoder-pred-kernel-size 9 \
            --encoder-pred-num-layers 8 \
            --encoder-pred-loss-scale 1.0 \
            --encoder-pred-logp-scale 0.0 \
            --encoder-pred-logp-ratio-clamp 0.0 \
            --encoder-pred-enc-in-rnn 0 \
            --encoder-pred-enc-in-raw 1 \
            --encoder-pred-dec-in-rnn 1 \
            --encoder-pred-dec-in-raw 1 \
            --encoder-pred-noise 0.1 \
            --encoder-pred-flow-depth 2 \
            --encoder-pred-flow-num-blocks 2 \
            --encoder-pred-flow-hidden-dim 320 \
            --max-duration 300 \
            --decoding-method $m 
      done
    done
  done
done