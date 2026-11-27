export SCENE="corridor"
export MODEL_NAME="kugamaxx/ras-diff"
export DATASET_NAME="kugamaxx/smokepv-control"

python3 ../run_evaluation_rasdiff.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --revision "scene/$SCENE" \
  --dataset_name_or_path $DATASET_NAME \
  --dataset_revision "scene/$SCENE" \
  --top_k 3 \
  --index_dim 1024
