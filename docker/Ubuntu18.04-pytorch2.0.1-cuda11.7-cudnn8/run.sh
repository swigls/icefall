docker run \
  --runtime=nvidia \
  --shm-size=2gb \
  --name=icefall \
  --gpus all \
  --rm \
  -it \
  -e HOME \
  --user $UID \
  -v $HOME:$HOME \
  --ipc=host \
  -w $PWD \
  icefall/pytorch2.0.1
#  --expose 6006 \