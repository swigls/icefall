export CUDA_VISIBLE_DEVICES="0,1,3,4"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --world-size 4 \
  --start-epoch 1 \
  --num-epochs 30 \
  --full-libri 1 \
  --use-fp16 1 \
  --enable-musan 0 \
  --max-duration 750 \
  --enable-spec-aug 0 \
  --manifest-dir data/ecemb_6.0 \
  --exp-dir $dir/exp-6.0-noSA \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --master-port 12535
