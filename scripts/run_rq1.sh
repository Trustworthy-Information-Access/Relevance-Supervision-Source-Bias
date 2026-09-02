#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/evaluation.local.yaml}"
MODEL_CONFIG="${2:-configs/models.yaml}"
DATASET="${3:-scifact}"
MODEL="${4:-ance}"

rsb retrieve --config "$CONFIG" --models "$MODEL_CONFIG" --dataset "$DATASET" --model "$MODEL"
