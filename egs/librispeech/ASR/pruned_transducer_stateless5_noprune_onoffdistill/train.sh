export CUDA_VISIBLE_DEVICES="0,1,2,3,4"

dir=$( dirname -- "$0"; )
./$dir/train.py \
  --master-port 12345 \
  --world-size 5 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 1 \
  --enable-musan 0 \
  --exp-dir $dir/exp \
  --max-duration 100 \
  --use-fp16 1 \
  --num-encoder-layers 24 \
  --dim-feedforward 1536 \
  --nhead 8 \
  --encoder-dim 384 \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --dynamic-chunk-training 1 \
  --causal-convolution 1 \
  --short-chunk-size 1 \
  --num-left-chunks 64 \
  --pronouncer-stop-gradient 0 \
  --loss-off-scale 1.0 \
  --loss-on-scale 1.0 \
  --loss-r-scale 1.0