export CUDA_VISIBLE_DEVICES="0,1,2,3,4"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --world-size 5 \
  --start-epoch 4 \
  --num-epochs 30 \
  --full-libri 1 \
  --use-fp16 1 \
  --enable-musan 0 \
  --max-duration 750 \
  --enable-spec-aug 1 \
  --manifest-dir data/fbank_base \
  --exp-dir $dir/exp-fbank_base_specorig \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --master-port 12535
