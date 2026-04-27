#!/bin/bash
# Train GeoRT IK model for LinkerHand L20 right hand.
# Usage: bash train_l20.sh [--w_chamfer 80] [--w_curvature 0.1] [--w_pinch 1.0] [--tag manus_l20_fullhand]
set -e

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

TAG="${TAG:-manus_l20_fullhand}"
HUMAN_DATA="${HUMAN_DATA:-manus_data}"

python geort/trainer.py \
    -hand linkerhand_l20_right \
    -human_data "${HUMAN_DATA}" \
    -ckpt_tag "${TAG}" \
    --w_chamfer "${W_CHAMFER:-80.0}" \
    --w_curvature "${W_CURVATURE:-0.1}" \
    --w_collision 0.0 \
    --w_pinch "${W_PINCH:-1.0}" \
    "$@"
