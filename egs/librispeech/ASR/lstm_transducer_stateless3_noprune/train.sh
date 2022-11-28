export CUDA_VISIBLE_DEVICES="1,2,3,4,5"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --world-size 5 \
  --num-epochs 40 \
  --start-epoch 38 \
  --exp-dir $dir/exp \
  --full-libri 1 \
  --use-fp16 1 \
  --enable-musan 0 \
  --max-duration 250 \
  --master-port 12325 \
  --num-encoder-layers 12 \
  --grad-norm-threshold 25.0 \
  --rnn-hidden-size 1024