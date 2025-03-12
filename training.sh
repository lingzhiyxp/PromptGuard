source /home/lingzhiyuan/anaconda3/etc/profile.d/conda.sh

SAFE_EMBEDDINGS="./safe_embeddings/YOUR_SAFE_EMBEDDINGS_FOLDER"
CUDA_DEVICE=1

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" accelerate launch --main_process_port 29501 ./scripts/our_pipeline_sd_sdedit_v2.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="TRAINING_IMAGE_FOLDER" \
  --edited_data_dir="EDITED_TOXIC_IMAGE_FOLDER" \
  --train_data_csv="THE_CORRESPONDING_CSV_FILE" \
  --placeholder_token="<prompt_guard>" \
  --initializer_token="safe" \
  --position="end" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="$SAFE_EMBEDDINGS" \
  --num_vectors=1 \
  --coefficient=0.1 \
  --resume_from_checkpoint "latest" \