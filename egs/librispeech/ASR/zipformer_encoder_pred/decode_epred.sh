dir=$( dirname -- "$0"; )

#for m in greedy_search; do
for m in greedy_search; do
  for epoch in 40; do
    for avg in 1; do
          CUDA_VISIBLE_DEVICES="1" ./$dir/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --causal 1 \
            --num-workers 8 \
            --chunk-size 8 \
            --left-context-frames 128 \
            --exp-dir $dir/exp-chunk8-dur600-eval-from25-pred-d-nl-lyr1-nopreact-encdecrnn \
            --rnnt-type "regular" \
            --train-in-eval-mode 1 \
            --use-encoder-pred 1 \
            --encoder-pred-detach 1 \
            --encoder-pred-bottleneck-dim 512 \
            --encoder-pred-kernel-size 17 \
            --encoder-pred-num-layers 1 \
            --encoder-pred-l2-norm-loss 1 \
            --encoder-pred-loss-scale 0.25 \
            --encoder-pred-l2-to-logp Gaussian \
            --encoder-pred-logp-scale 0.0 \
            --encoder-pred-logp-ratio-clamp 0.0 \
            --encoder-pred-enc-in-rnn 1 \
            --encoder-pred-dec-in-rnn 1 \
            --max-duration 600 \
            --decoding-method $m 
    done
  done
done