export SCENE="corridor"
export MODEL_NAME="kugamaxx/ras-diff"
export DATABASE_NAME="kugamaxx/smokepv-control"
export VALIDATION_CASE="validation/corridor_s00_c04_h1618"

python3 ../run_pipeline_rasdiff.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --revision "scene/$SCENE" \
  --database_name_or_path $DATABASE_NAME \
  --database_revision "scene/$SCENE" \
  --validation_case $VALIDATION_CASE \
  --top_k 3 \
  --index_dim 1024
