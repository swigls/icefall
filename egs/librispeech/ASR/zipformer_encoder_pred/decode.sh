dir=$( dirname -- "$0"; )

for m in beam_search; do
  for epoch in 68; do
    for avg in 10; do
      for logp_scale in 1; do
          CUDA_VISIBLE_DEVICES="1" ./$dir/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --causal 1 \
            --num-workers 8 \
            --chunk-size 8 \
            --left-context-frames 128 \
            --exp-dir $dir/exp-chunk8-dur600-eval-from25-pred-d-nl-from50-lps50000-nodetach \
            --rnnt-type "regular" \
            --train-in-eval-mode 1 \
            --use-encoder-pred 1 \
            --encoder-pred-detach 0 \
            --encoder-pred-bottleneck-dim 512 \
            --encoder-pred-kernel-size 17 \
            --encoder-pred-num-layers 2 \
            --encoder-pred-l2-norm-loss 1 \
            --encoder-pred-loss-scale 0.25 \
            --encoder-pred-l2-to-logp Gaussian \
            --encoder-pred-logp-scale $logp_scale \
            --encoder-pred-logp-ratio-clamp 0.0 \
            --max-duration 600 \
            --decoding-method $m 
      done
    done
  done
done