#! /bin/bash
#SBATCH --job-name=bitnet-train
#SBATCH -p sv-dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=250
#SBATCH --gres=gpu:8
#SBATCH --output=%x-%j_stdout.out
#SBATCH --error=%x-%j_err.out

export NCCL_DEBUG=INFO
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c '/home/takatoshi.sato/venv/bitnet/bin/python -m torch.distributed.run --nproc_per_node 8 run_clm.py \
    --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID --master_addr=$MASTER_ADDR  \
    --dataset_name='wikitext' \
    --dataset_config_name='wikitext-103-v1' \
    --model_type='llama' \
    --config_name='./bitllama-110M-config' \
    --tokenizer_name='beomi/llama-2-ko-7b' \
    --num_train_epochs=10 \
    --block_size=2048 \
    --per_device_train_batch_size=12 \
    --gradient_accumulation_steps=3 \
    --optim adafactor \
    --learning_rate=8e-4 \
    --torch_dtype bfloat16 \
    --bf16 \
    --output_dir='bitllama-wikitext' \
    --do_train \
    --save_strategy='epoch' \
    --logging_strategy='steps' \
    --logging_first_step \
    --logging_steps=10 \
    --save_total_limit=1 \
    --run_name='bitllama-wikitext' \
    --overwrite_output_dir \
    --report_to='mlflow' \
    --low_cpu_mem_usage \
    --save_steps 1000 \
    --ddp_timeout 360000 \
    --deepspeed config/ds_config.json'

