export CUDA_VISIBLE_DEVICES="1,2,3,4,5"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --world-size 5 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 1 \
  --use-fp16 1 \
  --enable-musan 1 \
  --exp-dir $dir/exp-troff \
  --num-left-chunks 4 \
  --short-chunk-size 50 \
  --short-chunk-threshold 0.0 \
  --max-duration 600 \
  --master-port 12345
  
#  --num-encoder-layers "2,2,2,2,2" \
#  --feedforward-dims "768,768,768,768,768" \
#  --encoder-dims "256,256,256,256,256" \
#  --encoder-unmasked-dims "192,192,192,192,192" \