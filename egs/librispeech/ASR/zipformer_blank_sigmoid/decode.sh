dir=$( dirname -- "$0"; )

#for m in greedy_search fast_beam_search modified_beam_search ; do
for m in modified_beam_search ; do
  for epoch in 40; do
    for avg in 13; do
        CUDA_VISIBLE_DEVICES="5" ./$dir/decode.py \
          --epoch $epoch \
          --avg $avg \
          --use-averaged-model 1 \
          --exp-dir ./$dir/exp-small \
          --blank-sigmoid 1 \
          --num-workers 8 \
          --max-duration 1000 \
          --decoding-method $m \
          --num-encoder-layers 2,2,2,2,2,2 \
          --feedforward-dim 512,768,768,768,768,768 \
          --encoder-dim 192,256,256,256,256,256 \
          --encoder-unmasked-dim 192,192,192,192,192,192
    done
  done
done