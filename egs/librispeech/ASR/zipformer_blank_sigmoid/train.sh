dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" ./$dir/train.py \
  --world-size 6 \
  --start-epoch 1 \
  --num-epochs 40 \
  --full-libri 1 \
  --num-workers 0 \
  --use-fp16 1 \
  --causal 0 \
  --blank-sigmoid 1 \
  --exp-dir $dir/exp-small \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,768,768,768,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --base-lr 0.04 \
  --max-duration 1000 \
  --master-port 12345
