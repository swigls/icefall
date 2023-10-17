dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" ./$dir/train.py \
  --world-size 6 \
  --start-epoch 1 \
  --num-epochs 50 \
  --causal 1 \
  --full-libri 1 \
  --use-fp16 1 \
  --exp-dir $dir/exp-paper-nl8-dim320-noise0.1-ls0.1-flow-dep2-dim320 \
  --chunk-size "8" \
  --rnnt-type "regular" \
  --prune-range 10 \
  --simple-loss-scale 0.5 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 768,768,768,768,768,768 \
  --encoder-dim 256,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --train-in-eval-mode 0 \
  --use-encoder-pred 1 \
  --enable-spec-aug 0 \
  --encoder-pred-detach 1 \
  --encoder-pred-bottleneck-dim 320 \
  --encoder-pred-kernel-size 9 \
  --encoder-pred-num-layers 8 \
  --encoder-pred-loss-scale 0.1 \
  --encoder-pred-logp-scale 0.0 \
  --encoder-pred-logp-ratio-clamp 0.0 \
  --encoder-pred-enc-in-rnn 0 \
  --encoder-pred-enc-in-raw 1 \
  --encoder-pred-dec-in-rnn 1 \
  --encoder-pred-dec-in-raw 1 \
  --encoder-pred-noise 0.1 \
  --encoder-pred-flow-depth 2 \
  --encoder-pred-flow-num-blocks 2 \
  --encoder-pred-flow-hidden-dim 320 \
  --max-duration 400 \
  --master-port 12345
  
#  --encoder-pred-
#  --exp-dir $dir/exp \
#  --max-duration 750 \
#  --base-lr 0.045 \
