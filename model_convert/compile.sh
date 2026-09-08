#!/bin/bash
# Quantize + compile the exported graphs for AX650/NPU3.
#
# Requires Docker and the Pulsar2 image used for FS-EEND.AXERA.
# Run from model_convert/ after export_models.py and generate_calibration.py.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${PULSAR2_IMAGE:-docker-registry.aitsw.axera-tech.com/pulsar2:20260810-temp-09cadfa9}"

[ -f export/segmentation_sincnet.onnx ] || { echo "missing export/segmentation_sincnet.onnx"; exit 1; }
[ -f export/segmentation_lstm.onnx ]    || { echo "missing export/segmentation_lstm.onnx"; exit 1; }
[ -f export/embedding.onnx ]            || { echo "missing export/embedding.onnx"; exit 1; }
[ -f calib_data/waveform.tar.gz ]       || { echo "missing calib_data/*.tar.gz (run generate_calibration.py + pack_calib.sh)"; exit 1; }

for cfg in pulsar2_config_seg_sincnet pulsar2_config_seg_lstm pulsar2_config_emb; do
    docker run --rm -v "$PWD":/workspace -w /workspace "$IMAGE" \
        pulsar2 build --config "/workspace/$cfg.json"
done

echo "built: compile/segmentation_sincnet.axmodel compile/segmentation_lstm.axmodel compile/embedding.axmodel"
