dir=$( dirname -- "$0"; )

#for m in greedy_search modified_beam_search; do
for m in greedy_search; do
  for epoch in 18; do
    for avg in 1; do
          CUDA_VISIBLE_DEVICES="4" ./$dir/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --causal 1 \
            --num-workers 8 \
            --chunk-size 8 \
            --left-context-frames 256 \
            --exp-dir $dir/exp-small \
            --rnnt-type "regular" \
            --num-encoder-layers 2,2,2,2,2,2 \
            --feedforward-dim 512,768,768,768,768,768 \
            --encoder-dim 192,256,256,256,256,256 \
            --encoder-unmasked-dim 192,192,192,192,192,192 \
            --train-in-eval-mode 1 \
            --use-encoder-pred 0 \
            --max-duration 600 \
            --decoding-method $m 
    done
  done
done