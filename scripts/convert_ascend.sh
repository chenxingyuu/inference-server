#!/usr/bin/env bash
# Convert an ONNX model to multiple Ascend .om files (one per batch size).
# Usage: ./scripts/convert_ascend.sh <onnx_path> <model_name> [soc_version]
# Example: ./scripts/convert_ascend.sh yolo11s.onnx yolo11s Ascend310P3
set -euo pipefail

ONNX="${1:?Usage: $0 <onnx_path> <model_name> [soc_version]}"
NAME="${2:?}"
SOC="${3:-Ascend310P3}"
OUT_DIR="$(dirname "$ONNX")"

for BS in 1 4 8 16; do
    echo "Converting batch=${BS} ..."
    atc \
        --model="${ONNX}" \
        --framework=5 \
        --output="${OUT_DIR}/${NAME}_b${BS}" \
        --soc_version="${SOC}" \
        --input_shape="images:${BS},3,640,640" \
        --precision_mode=allow_mix_precision \
        --log=error
    echo "  → ${OUT_DIR}/${NAME}_b${BS}.om"
done

echo "Done. Generated ${OUT_DIR}/${NAME}_b{1,4,8,16}.om"
