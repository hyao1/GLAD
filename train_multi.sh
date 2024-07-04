export INSTANCE_DIR='/hdd/Datasets/VisA'
export OUTPUT_DIR="20000step_bs32_eps_anomaly2_multiclass"
export ANOMALY_DIR="/hdd/Datasets/dtd"
export DENOISE_STEP=500
export MAX_TRAIN_STEP=20000
export SEED=0

accelerate launch main_multi.py \
    --train=True \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --anomaly_data_dir=$ANOMALY_DIR \
    --checkpointing_steps=10000 \
    --denoise_step=$DENOISE_STEP \
    --instance_prompt="a photo of sks" \
    --resolution=256 \
    --train_batch_size=32 \
    --gradient_accumulation_steps=1 --gradient_checkpointing \
    --use_8bit_adam \
    --set_grads_to_none \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=$MAX_TRAIN_STEP \
    --pre_compute_text_embeddings \
    --seed=$SEED

