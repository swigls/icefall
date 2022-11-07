export CUDA_VISIBLE_DEVICES="0,1,2,3"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --world-size 4 \
  --num-epochs 100 \
  --start-epoch 31 \
  --exp-dir $dir/exp \
  --full-libri 1 \
  --use-fp16 1 \
  --enable-musan 0 \
  --max-duration 150 \
  --master-port 12325 \
  --num-encoder-layers 12 \
  --grad-norm-threshold 25.0 \
  --rnn-hidden-size 1024 \
  --kmeans-model $dir/exp/kmeans_500.npy \
  --pronouncer-stop-gradient 0 \
  --pronouncer-lambda 1.0 \
  --pronouncer-normalize 0
