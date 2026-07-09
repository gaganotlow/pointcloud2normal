#!/bin/bash
# Train MSECNet + CNN/ResNet RGB fusion.

set -e

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")

LABELS="${LABELS:-$ROOT/shared/normal_labels_patch03.npz}"
PCD_DIR="${PCD_DIR:-$ROOT/data/pcd_dataset_roi}"
IMAGE_DIR="${IMAGE_DIR:-$ROOT/data/yolo_seg_by_car}"
RGB_BACKBONE="${RGB_BACKBONE:-resnet50}"
RESNET_STAGE="${RESNET_STAGE:-layer2}"
RGB_PRETRAINED="${RGB_PRETRAINED:-1}"
FREEZE_RGB_BACKBONE_STEPS="${FREEZE_RGB_BACKBONE_STEPS:-5000}"
RGB_BACKBONE_LR_MULT="${RGB_BACKBONE_LR_MULT:-0.05}"
OUTPUT_DIR_PREFIX="${OUTPUT_DIR_PREFIX:-$HERE/ckpt_${RGB_BACKBONE}}"
OUTPUT_AUTO=0
if [ -z "${OUTPUT_DIR+x}" ]; then
    OUTPUT_AUTO=1
    RUN_ID=1
    while true; do
        OUTPUT_DIR=$(printf "%s_%03d" "$OUTPUT_DIR_PREFIX" "$RUN_ID")
        if [ ! -e "$OUTPUT_DIR" ]; then
            mkdir -p "$(dirname "$OUTPUT_DIR")"
            mkdir "$OUTPUT_DIR"
            break
        fi
        RUN_ID=$((RUN_ID + 1))
    done
fi

STEPS="${STEPS:-30000}"
BS="${BS:-8}"
MAX_POINTS="${MAX_POINTS:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
CNN_DIM="${CNN_DIM:-128}"
RGB_FEAT_DIM="${RGB_FEAT_DIM:-512}"
AUG_DEG="${AUG_DEG:-0}"
LR="${LR:-2e-4}"
VAL_EVERY="${VAL_EVERY:-1000}"
SOFT="${SOFT:---soft}"
PRETRAINED_FLAG=""
if [ "$RGB_PRETRAINED" = "0" ]; then
    PRETRAINED_FLAG="--no-rgb-pretrained"
fi

echo "=========================================="
echo "MSECNet + CNN/ResNet RGB training"
echo "=========================================="
echo "labels:     $LABELS"
echo "pcd_dir:    $PCD_DIR"
echo "image_dir:  $IMAGE_DIR"
echo "output:     $OUTPUT_DIR"
echo "auto_out:   $OUTPUT_AUTO"
echo "image_size: $IMAGE_SIZE"
echo "backbone:   $RGB_BACKBONE"
echo "stage:      $RESNET_STAGE"
echo "pretrained: $RGB_PRETRAINED"
echo "bb_lr_mult: $RGB_BACKBONE_LR_MULT"
echo ""

python "$HERE/train_cnn_rgb.py" \
    "$LABELS" \
    "$PCD_DIR" \
    --image-dir "$IMAGE_DIR" \
    $SOFT \
    --steps "$STEPS" \
    --bs "$BS" \
    --max-points "$MAX_POINTS" \
    --image-size "$IMAGE_SIZE" \
    --cnn-dim "$CNN_DIM" \
    --rgb-feat-dim "$RGB_FEAT_DIM" \
    --rgb-backbone "$RGB_BACKBONE" \
    --resnet-stage "$RESNET_STAGE" \
    $PRETRAINED_FLAG \
    --freeze-rgb-backbone-steps "$FREEZE_RGB_BACKBONE_STEPS" \
    --rgb-backbone-lr-mult "$RGB_BACKBONE_LR_MULT" \
    --aug-deg "$AUG_DEG" \
    --lr "$LR" \
    --val-every "$VAL_EVERY" \
    --out "$OUTPUT_DIR"

echo ""
echo "done -> $OUTPUT_DIR/best.pt"
