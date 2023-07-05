dir=$( dirname -- "$0"; )

for m in greedy_search; do
  for epoch in 4; do
    for avg in 2; do
          CUDA_VISIBLE_DEVICES="0" ./$dir/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --exp-dir ./$dir/exp-chunk8 \
            --causal 1 \
            --num-workers 8 \
            --chunk-size 8 \
            --left-context-frames 128 \
            --use-encoder-pred 0 \
            --encoder-pred-loss-scale 0.1 \
            --encoder-pred-bottleneck-dim 384 \
            --encoder-pred-kernel-size 17 \
            --encoder-pred-num-layers 2 \
            --encoder-pred-logp-scale 0.0 \
            --encoder-pred-detach 0 \
            --max-duration 100 \
            --decoding-method $m 
    done
  done
done