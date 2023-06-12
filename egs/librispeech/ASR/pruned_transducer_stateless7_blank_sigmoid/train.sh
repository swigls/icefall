dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" ./$dir/train.py \
  --world-size 6 \
  --start-epoch 5 \
  --num-epochs 30 \
  --full-libri 1 \
  --use-fp16 0 \
  --base-lr 0.05 \
  --max-duration 275 \
  --blank-sigmoid 1 \
  --exp-dir $dir/exp \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --master-port 12345
