dir=$( dirname -- "$0"; )

for decoding_method in greedy_search modified_beam_search modified_beam_search_joint; do
#for decoding_method in modified_beam_search_joint; do
#for decoding_method in greedy_search_joint; do
#for decoding_method in beam_search beam_search_joint; do
  ./$dir/decode.py \
    --epoch 1 \
    --avg 1 \
    --use-averaged-model False \
    --exp-dir $dir/exp-normsgt0 \
    --max-duration 300 \
    --num-encoder-layers 12 \
    --rnn-hidden-size 1024 \
    --decoding-method $decoding_method \
    --kmeans-model $dir/exp/kmeans_500.npy \
    --pronouncer-stop-gradient 0 \
    --pronouncer-lambda 1.0 \
    --pronouncer-normalize 1 \
    --max-sym-per-frame 4 \
    --beam 8 \
    --max-contexts 4 \
    --max-states 8 \
    --beam-size 8
done
