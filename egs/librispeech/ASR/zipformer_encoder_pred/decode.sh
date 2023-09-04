dir=$( dirname -- "$0"; )

for m in greedy_search; do
  for epoch in 38; do
    for avg in 1; do
      for logp_scale in 0; do
          CUDA_VISIBLE_DEVICES="0" ./$dir/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --causal 1 \
            --num-workers 8 \
            --chunk-size 8 \
            --left-context-frames 128 \
            --exp-dir $dir/exp-chunk8base-from25-pred-d-lyr2-decrawrnn-from33-lps32 \
            --rnnt-type "regular" \
            --train-in-eval-mode 1 \
            --use-encoder-pred 1 \
            --encoder-pred-detach 1 \
            --encoder-pred-bottleneck-dim 512 \
            --encoder-pred-kernel-size 17 \
            --encoder-pred-num-layers 2 \
            --encoder-pred-l2-norm-loss 0 \
            --encoder-pred-loss-scale 1.0 \
            --encoder-pred-l2-to-logp Gaussian \
            --encoder-pred-logp-scale 32 \
            --encoder-pred-logp-ratio-clamp 0.0 \
            --encoder-pred-enc-in-rnn 0 \
            --encoder-pred-dec-in-rnn 1 \
            --encoder-pred-dec-in-raw 1 \
            --max-duration 600 \
            --decoding-method $m 
      done
    done
  done
done