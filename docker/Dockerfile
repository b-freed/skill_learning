FROM lambda-stack:18.04
FROM python:3.8.10

WORKDIR /root

COPY requirements.txt scripts/setup.sh scripts/set_vars.sh ./

RUN chmod +x setup.sh set_vars.sh && . ./setup.sh

ENV WS_PATH=/root
ENV MUJOCO_PY_MUJOCO_PATH=$WS_PATH/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WS_PATH/.mujoco/mujoco210/bin
ENV DISPLAY=:1

RUN apt-get update && apt-get install -y libosmesa6-dev build-essential patchelf \
ffmpeg libgl1-mesa-dev python-opengl apt-utils curl xvfb xorg-dev libsdl2-dev swig cmake

RUN pip install -r requirements.txt
RUN python -c "import mujoco_py"

COPY docker/render_start.sh ./

ENV D4RL_SUPPRESS_IMPORT_ERROR=1

CMD . ./render_start.sh
