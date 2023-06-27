dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" ./$dir/train.py \
  --world-size 6 \
  --start-epoch 1 \
  --num-epochs 40 \
  --full-libri 1 \
  --use-fp16 1 \
  --causal 1 \
  --exp-dir $dir/exp \
  --max-duration 750 \
  --master-port 12345
  
#  --exp-dir $dir/exp \
#  --max-duration 750 \
#  --base-lr 0.045 \
