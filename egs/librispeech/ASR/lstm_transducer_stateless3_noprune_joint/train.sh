export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --world-size 7 \
  --num-epochs 40 \
  --start-epoch 2 \
  --exp-dir $dir/exp-normsgt0 \
  --full-libri 1 \
  --use-fp16 1 \
  --enable-musan 0 \
  --max-duration 80 \
  --master-port 12325 \
  --num-encoder-layers 12 \
  --grad-norm-threshold 25.0 \
  --rnn-hidden-size 1024 \
  --kmeans-model $dir/exp/kmeans_500.npy \
  --pronouncer-stop-gradient 0 \
  --pronouncer-lambda 1.0 \
  --pronouncer-normalize 1
