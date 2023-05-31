dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3"
./$dir/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --full-libri 1 \
  --use-fp16 1 \
  --max-duration 280 \
  --exp-dir $dir/exp \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --master-port 12535
