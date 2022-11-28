export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --world-size 7 \
  --num-epochs 40 \
  --start-epoch 1 \
  --exp-dir $dir/exp-normsg-lam0.01 \
  --full-libri 1 \
  --use-fp16 1 \
  --enable-musan 0 \
  --max-duration 80 \
  --master-port 12325 \
  --num-encoder-layers 12 \
  --grad-norm-threshold 25.0 \
  --rnn-hidden-size 1024 \
  --pronouncer-stop-gradient 0 \
  --pronouncer-lambda 0.01 \
  --pronouncer-normalize 1