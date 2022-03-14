#!/bin/sh

# Download mujoco binaries for linux and deepmind free key
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
wget https://www.roboti.us/file/mjkey.txt

# Move files to local mujoco directory
mkdir .mujoco
tar -zxf mujoco210-linux-x86_64.tar.gz --directory=.mujoco
mv mjkey.txt ./.mujoco
rm mujoco210-linux-x86_64.tar.gz

. ./set_vars.sh

# Also need the following if not already installed on the system for mujoco to work
# sudo apt install libosmesa6-dev
# sudo apt install build-essential
# sudo apt install patchelf
