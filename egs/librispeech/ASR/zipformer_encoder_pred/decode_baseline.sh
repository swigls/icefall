dir=$( dirname -- "$0"; )

#for m in greedy_search modified_beam_search; do
for m in greedy_search; do
  for epoch in 50; do
    for avg in 15; do
          CUDA_VISIBLE_DEVICES="1" ./$dir/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --causal 1 \
            --num-workers 8 \
            --chunk-size 8 \
            --left-context-frames 128 \
            --exp-dir $dir/exp-100h-eval \
            --train-in-eval-mode 1 \
            --use-encoder-pred 0 \
            --encoder-pred-detach 0 \
            --encoder-pred-bottleneck-dim 384 \
            --encoder-pred-kernel-size 17 \
            --encoder-pred-num-layers 2 \
            --encoder-pred-loss-scale 0.25 \
            --encoder-pred-l2-norm-loss 0 \
            --encoder-pred-l2-to-logp Gaussian \
            --encoder-pred-logp-scale 1.0 \
            --encoder-pred-logp-ratio-clamp 0.0 \
            --max-duration 100 \
            --decoding-method $m 
            #--exp-dir $dir/exp-chunk8-dur600-eval \
    done
  done
done