dir=$( dirname -- "$0"; )

for m in greedy search; do
  for epoch in 30; do
    for avg in 8; do
          CUDA_VISIBLE_DEVICES="0" ./$dir/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --exp-dir ./$dir/exp \
            --causal 1 \
            --num-workers 8 \
            --chunk-size 16 \
            --left-context-frames 128 \
            --max-duration 1000 \
            --decoding-method $m 
    done
  done
done