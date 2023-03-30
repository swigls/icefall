export CUDA_VISIBLE_DEVICES="1,2,3,4,5"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --world-size 5 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 1 \
  --use-fp16 1 \
  --enable-musan 1 \
  --exp-dir $dir/exp \
  --num-left-chunks 4 \
  --short-chunk-size 50 \
  --short-chunk-threshold 0.75 \
  --max-duration 300 \
  --master-port 12345
