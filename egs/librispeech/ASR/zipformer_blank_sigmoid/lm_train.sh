dir=$( dirname -- "$0"; )

CUDA_VISIBLE_DEVICES="0,1,2,3"
../../../icefall/rnn_lm/train.py \
    --start-epoch 44 \
    --world-size 4 \
    --num-epochs 100 \
    --exp-dir $dir/exp-rnnlm \
    --use-fp16 0 \
    --tie-weights 0 \
    --embedding-dim 800 \
    --hidden-dim 200 \
    --num-layers 2 \
    --batch-size 400
