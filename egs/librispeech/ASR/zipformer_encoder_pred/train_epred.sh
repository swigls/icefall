dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" ./$dir/train.py \
  --world-size 1 \
  --start-epoch 51 \
  --num-epochs 100 \
  --causal 1 \
  --full-libri 1 \
  --use-fp16 1 \
  --exp-dir $dir/exp-small-from50-encraw-d-nl8-ls0.01-decrawrnn-flow-dep2-dim320-pr5-sls0.1 \
  --chunk-size "8" \
  --rnnt-type "regular" \
  --prune-range 5 \
  --simple-loss-scale 0.1 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,768,768,768,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --train-in-eval-mode 1 \
  --use-encoder-pred 1 \
  --encoder-pred-detach 1 \
  --encoder-pred-bottleneck-dim 320 \
  --encoder-pred-kernel-size 9 \
  --encoder-pred-num-layers 8 \
  --encoder-pred-loss-scale 0.01 \
  --encoder-pred-logp-scale 0.0 \
  --encoder-pred-logp-ratio-clamp 0.0 \
  --encoder-pred-enc-in-rnn 0 \
  --encoder-pred-enc-in-raw 1 \
  --encoder-pred-dec-in-rnn 1 \
  --encoder-pred-dec-in-raw 1 \
  --encoder-pred-flow-depth 2 \
  --encoder-pred-flow-num-blocks 2 \
  --encoder-pred-flow-hidden-dim 320 \
  --max-duration 600 \
  --master-port 12345
  
#  --encoder-pred-
#  --exp-dir $dir/exp \
#  --max-duration 750 \
#  --base-lr 0.045 \
