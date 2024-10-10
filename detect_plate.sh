#!/bin/sh

export DISPLAY=:0
#python /workspace/truck/detect_plate.py --source 'v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,fps=30/1 ! videoconvert ! appsink'
# Intake B6
python /workspace/truck/detect_plate.py --source 'rtspsrc location=rtsp://admin:admin1234@10.1.26.228:554/ISAPI/Streaming/Channels/1402/picture latency=150 ! queue ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! appsink'
# truck scale
#python /workspace/truck/detect_plate.py --source 'rtspsrc location=rtsp://admin:admin1234@10.1.26.222:554/ISAPI/Streaming/Channels/102/picture latency=150 ! queue ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! appsink'
