#!/usr/bin/env bash
set -euo pipefail

# Copy configs/evaluation.example.yaml and configs/matrix.example.yaml to local
# files, change the paths, and then run or resume the complete matrix.
rsb run-matrix --config configs/matrix.local.yaml
