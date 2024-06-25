export INSTANCE_DIR='/hdd/Datasets/MVTec-AD'
export ANOMALY_DIR="/hdd/Datasets/dtd"
export CLASS_NAME="all"
export SEED=0

accelerate launch main.py \
    --instance_data_dir=$INSTANCE_DIR \
    --anomaly_data_dir=$ANOMALY_DIR \
    --checkpointing_steps=500 \
    --instance_prompt="a photo of sks" \
    --resolution=512 \
    --test_batch_size=2 \
    --class_name=$CLASS_NAME \
    --pre_compute_text_embeddings \
    --seed=$SEED