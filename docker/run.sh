#!/bin/sh

TAG='deepstreamv7'

xhost +local:root
docker run -dit \
  --device=/dev/video0:/dev/video0\
  --restart always \
  --runtime nvidia \
  --net=host \
  --name jetson \
  ezvk7740/jetson:deepstreamv7
xhost -local:root
