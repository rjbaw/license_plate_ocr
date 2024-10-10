#!/bin/sh

sudo addgroup --system camera_boot
sudo adduser --system --no-create-home --disabled-login --disabled-password --ingroup camera_boot camera_boot
sudo adduser camera_boot video
sudo adduser camera_boot docker
sudo cp camera_boot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start camera_boot
sudo systemctl status camera_boot
sudo systemctl enable camera_boot
chmod +x run.sh
sh run.sh
