export CUDA_VISIBLE_DEVICES="1,2,3,4,5"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --world-size 5 \
  --num-epochs 100 \
  --start-epoch 31 \
  --full-libri 0 \
  --use-fp16 1 \
  --enable-musan 1 \
  --exp-dir $dir/exp-100 \
  --num-left-chunks 4 \
  --short-chunk-size 50 \
  --short-chunk-threshold 0.75 \
  --max-duration 260 \
  --master-port 12345
