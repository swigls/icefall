dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" ./$dir/train.py \
  --world-size 6 \
  --start-epoch 71 \
  --num-epochs 100 \
  --causal 1 \
  --full-libri 1 \
  --use-fp16 1 \
  --exp-dir $dir/exp-small-from70-pr20 \
  --chunk-size "8" \
  --rnnt-type "regular" \
  --no-prune 0 \
  --prune-range 20 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,768,768,768,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --train-in-eval-mode 1 \
  --use-encoder-pred 0 \
  --encoder-pred-detach 1 \
  --encoder-pred-bottleneck-dim 512 \
  --encoder-pred-kernel-size 17 \
  --encoder-pred-num-layers 2 \
  --encoder-pred-l2-norm-loss 1 \
  --encoder-pred-loss-scale 0.25 \
  --encoder-pred-l2-to-logp Gaussian \
  --encoder-pred-logp-scale 0.0 \
  --encoder-pred-logp-ratio-clamp 0.0 \
  --max-duration 250 \
  --master-port 12345
  
#  --exp-dir $dir/exp \
#  --max-duration 750 \
#  --base-lr 0.045 \
