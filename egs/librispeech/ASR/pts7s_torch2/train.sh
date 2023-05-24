dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3"
./$dir/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir $dir/exp \
  --full-libri 1 \
  --max-duration 300 \
  --master-port 12345
