export INSTANCE_DIR='/hdd/Datasets/PCBBank'
export OUTPUT_DIR="20000step_bs32_eps_anomaly2_multiclass"
export SEED=0

accelerate launch main_multi.py \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --resolution=256 \
    --test_batch_size=2 \
    --pre_compute_text_embeddings \
    --seed=$SEED