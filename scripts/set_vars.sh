#!/bin/sh

# Set environment variables
export MUJOCO_PY_MJKEY_PATH=$(pwd)'/.mujoco/mjkey.txt'
export MUJOCO_PY_MUJOCO_PATH=$(pwd)'/.mujoco/mujoco210'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)'/.mujoco/mujoco210/bin'
