#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-7B  # replace it with your local file path
TEACHER_MODEL=open-r1/OpenR1-Qwen-7B        # replace it with your local file path
format_prompt=./examples/format_prompt/no.jinja
reward_path=./examples/reward_function/math.py:compute_score

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.format_prompt=${format_prompt} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.use_mentor=True \
    worker.rollout.n=8 \
    worker.rollout.n_mixedpolicy=4 \
    worker.actor.entropy_p=0.95 \
    algorithm.expert_weight_decay_totalstep=120 \
    algorithm.expert_weight=1.0 \
    worker.rollout.teacher_model_path=${TEACHER_MODEL} \
    worker.reward.reward_function=${reward_path} \
    trainer.experiment_name=MENTOR \
    data.max_response_length=8192 \
    worker.rollout.max_num_batched_tokens=18440 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
