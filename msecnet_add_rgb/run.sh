#!/bin/bash
# 快速启动脚本：预计算MoGe特征 + 训练MSECNet+RGB

set -e

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")

# 配置路径
PCD_DIR="${PCD_DIR:-$ROOT/data/pcd_dataset_roi}"
LABELS="${LABELS:-$ROOT/shared/normal_labels_patch03.npz}"
MOGE_FEAT_DIR="${MOGE_FEAT_DIR:-$ROOT/data/moge_features}"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/ckpt}"

echo "=========================================="
echo "MSECNet + MoGe RGB特征融合训练"
echo "=========================================="
echo "点云数据: $PCD_DIR"
echo "标签文件: $LABELS"
echo "MoGe特征: $MOGE_FEAT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 步骤1: 检查MoGe特征是否已预计算
if [ ! -d "$MOGE_FEAT_DIR" ] || [ -z "$(ls -A $MOGE_FEAT_DIR 2>/dev/null)" ]; then
    echo "[步骤1] 预计算MoGe特征..."
    echo "请确保已激活MoGe环境，然后运行："
    echo ""
    echo "  python $HERE/precompute_moge_feat.py \\"
    echo "      $PCD_DIR \\"
    echo "      $MOGE_FEAT_DIR \\"
    echo "      --device cuda"
    echo ""
    echo "完成后重新运行本脚本。"
    exit 1
else
    FEAT_COUNT=$(ls -1 "$MOGE_FEAT_DIR"/*.npz 2>/dev/null | wc -l)
    echo "[步骤1] 已找到 $FEAT_COUNT 个预计算的MoGe特征，跳过预计算"
fi

# 步骤2: 训练MSECNet+RGB
echo ""
echo "[步骤2] 开始训练MSECNet+RGB..."
echo ""

# 默认训练参数
STEPS="${STEPS:-12000}"
BS="${BS:-12}"
MAX_POINTS="${MAX_POINTS:-0}"
AUG_DEG="${AUG_DEG:-45}"
LR="${LR:-5e-4}"
VAL_EVERY="${VAL_EVERY:-1000}"
SOFT="${SOFT:---soft}"

python "$HERE/train_rgb_fusion.py" \
    "$LABELS" \
    "$PCD_DIR" \
    "$MOGE_FEAT_DIR" \
    $SOFT \
    --steps $STEPS \
    --bs $BS \
    --max-points $MAX_POINTS \
    --aug-deg $AUG_DEG \
    --lr $LR \
    --val-every $VAL_EVERY \
    --out "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "训练完成！"
echo "最佳模型保存至: $OUTPUT_DIR/best.pt"
echo "=========================================="
