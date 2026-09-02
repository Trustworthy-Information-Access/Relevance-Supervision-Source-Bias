#!/usr/bin/env bash
set -euo pipefail

rsb train --config configs/training/rq3_in_batch_only.example.yaml
rsb train --config configs/training/rq3_standard.example.yaml
rsb train --config configs/training/rq3_hard_neg_only.example.yaml
