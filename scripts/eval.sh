export SCENE="corridor"
export MODEL_NAME="kugamaxx/ras-diff"
export DATABASE_NAME="kugamaxx/smokepv-control"
export VALIDATION_CASE="validation/corridor_s00_c04_h1618"

python3 ../run_evaluation_rasdiff.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --revision "scene/$SCENE" \
  --dataset_name_or_path $DATABASE_NAME \
  --dataset_revision "scene/$SCENE" \
  --top_k 3 \
  --index_dim 1024
