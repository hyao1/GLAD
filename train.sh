export INSTANCE_DIR='/hdd/Datasets/MVTec-AD'
export OUTPUT_DIR="4000step_bs32_eps_anomaly"
export ANOMALY_DIR="/hdd/Datasets/dtd"
export CLASS_NAME="all"
export MAX_TRAIN_STEP=4000
export SEED=0

accelerate launch main.py \
    --train=True \
    --instance_data_dir=$INSTANCE_DIR \
    --anomaly_data_dir=$ANOMALY_DIR \
    --output_dir=$OUTPUT_DIR \
    --checkpointing_steps=500 \
    --instance_prompt="a photo of sks" \
    --resolution=512 \
    --train_batch_size=32 \
    --class_name=$CLASS_NAME \
    --gradient_accumulation_steps=1 --gradient_checkpointing \
    --use_8bit_adam \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=$MAX_TRAIN_STEP \
    --pre_compute_text_embeddings \
    --seed=$SEED