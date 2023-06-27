dir=$( dirname -- "$0"; )

#for m in greedy_search fast_beam_search modified_beam_search ; do
#for m in modified_beam_search_LODR; do
for m in modified_beam_search_lm_shallow_fusion; do
#for m in modified_beam_search; do
  for epoch in 40; do
    for avg in 13; do
      #for lm_scale in 0.25 0.5 0.75 1.0; do
      #for lm_scale in 0.38 0.45 0.55; do
      for lm_scale in 1.0; do
        for ilm_scale in 0.0; do
          #ilm_scale=$(python -c "print(1 - $lm_scale)")
          sub_ilm_scale=$(python -c "print(-$lm_scale)")
          CUDA_VISIBLE_DEVICES="5" ./$dir/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model 1 \
            --exp-dir ./$dir/exp-small-plt \
            --blank-sigmoid 1 \
            --num-workers 8 \
            --max-duration 1000 \
            --decoding-method $m \
            --num-encoder-layers 2,2,2,2,2,2 \
            --feedforward-dim 512,768,768,768,768,768 \
            --encoder-dim 192,256,256,256,256,256 \
            --encoder-unmasked-dim 192,192,192,192,192,192 \
            --priorless-training 1 \
            --ilm-free-decoding 1 \
            --ilm-scale $ilm_scale \
            --sub-ilm-scale $sub_ilm_scale \
            --use-shallow-fusion 1 \
            --lm-type rnn \
            --lm-exp-dir ./$dir/icefall-librispeech-rnn-lm/exp \
            --lm-scale $lm_scale \
            --lm-epoch 99 \
            --lm-avg 1 \
            --rnn-lm-embedding-dim 2048 \
            --rnn-lm-hidden-dim 2048 \
            --rnn-lm-num-layers 3 \
            --rnn-lm-tie-weights 1 \
            --tokens-ngram 2 \
            --ngram-lm-scale -0.16
        done
      done
    done
  done
done