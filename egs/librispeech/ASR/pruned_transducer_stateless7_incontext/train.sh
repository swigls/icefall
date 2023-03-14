export CUDA_VISIBLE_DEVICES="0,1,2,3,4"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --world-size 5 \
  --num-epochs 30 \
  --full-libri 0 \
  --use-fp16 1 \
  --max-duration 600 \
  --exp-dir $dir/exp-warmup \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --master-port 12345
