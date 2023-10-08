dir=$( dirname -- "$0"; )

#for m in greedy_search modified_beam_search; do
for m in greedy_search; do
  for epoch in $(seq 92 100); do
    for avg in 1; do
          CUDA_VISIBLE_DEVICES="1" ./$dir/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --causal 1 \
            --num-workers 8 \
            --chunk-size 8 \
            --left-context-frames 128 \
            --exp-dir $dir/exp-small-from70-pr20 \
            --rnnt-type "regular" \
            --num-encoder-layers 2,2,2,2,2,2 \
            --feedforward-dim 512,768,768,768,768,768 \
            --encoder-dim 192,256,256,256,256,256 \
            --encoder-unmasked-dim 192,192,192,192,192,192 \
            --train-in-eval-mode 1 \
            --use-encoder-pred 0 \
            --encoder-pred-detach 1 \
            --encoder-pred-bottleneck-dim 512 \
            --encoder-pred-kernel-size 17 \
            --encoder-pred-num-layers 1 \
            --encoder-pred-l2-norm-loss 1 \
            --encoder-pred-loss-scale 0.25 \
            --encoder-pred-l2-to-logp Gaussian \
            --encoder-pred-logp-scale 0.0 \
            --encoder-pred-logp-ratio-clamp 0.0 \
            --max-duration 600 \
            --decoding-method $m 
    done
  done
done