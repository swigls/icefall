export CUDA_VISIBLE_DEVICES="1,2,3,4,6"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --world-size 5 \
  --start-epoch 10 \
  --num-epochs 30 \
  --full-libri 1 \
  --base-lr 0.01 \
  --print-diagnostics 0 \
  --use-fp16 1 \
  --short-chunk-threshold 0 \
  --short-chunk-size 50 \
  --max-duration 600 \
  --exp-dir $dir/exp \
  --master-port 12345
