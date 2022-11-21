dir=$( dirname -- "$0"; )

for decoding_method in greedy_search modified_beam_search modified_beam_search_joint; do
#for decoding_method in modified_beam_search_joint; do
#for decoding_method in greedy_search_joint; do
#for decoding_method in beam_search beam_search_joint; do
  ./$dir/decode.py \
    --epoch 21 \
    --avg 7 \
    --use-averaged-model True \
    --exp-dir $dir/exp-normsg-lam0.3 \
    --max-duration 300 \
    --num-encoder-layers 12 \
    --rnn-hidden-size 1024 \
    --decoding-method $decoding_method \
    --kmeans-model $dir/exp/kmeans_500.npy \
    --pronouncer-stop-gradient 0 \
    --pronouncer-lambda 0.3 \
    --pronouncer-normalize 1 \
    --max-sym-per-frame 4 \
    --beam 8 \
    --max-contexts 4 \
    --max-states 8 \
    --beam-size 8
done
