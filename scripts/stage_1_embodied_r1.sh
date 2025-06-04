set -x

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
RUN_NAME=embodiedr1_stage_1

python3 -m verl.trainer.main \
    config=scripts/config_stage1.yaml \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    data.max_prompt_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.n=8 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.limit_images=2 \
    worker.reward.score_function=embodiedr1 \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=2 \
    trainer.save_checkpoint_path=./workdir/${RUN_NAME} \
    trainer.val_freq=20 \
    trainer.save_freq=20 
