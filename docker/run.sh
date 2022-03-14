docker run --name skill_learning_env_$1 \
    -it \
    --rm \
    --gpus all \
    --mount type=bind,source="$(pwd)",target=/root/src \
    --dns 8.8.8.8 \
    thomasw219/lambda-mujoco-skill-learning:latest
