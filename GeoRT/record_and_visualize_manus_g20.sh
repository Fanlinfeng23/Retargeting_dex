#!/bin/bash
set -eo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/roslog_record_manus_g20}"
mkdir -p "$ROS_LOG_DIR"

NAME="${1:-manus_10s}"
DURATION="${DURATION:-10}"
FPS="${FPS:-30}"
GLOVE_ID="${GLOVE_ID:-0}"
PORT="${PORT:-8765}"
REUSE_BRIDGE="${REUSE_BRIDGE:-0}"

port_is_active() {
    timeout 1 bash -c "</dev/tcp/127.0.0.1/$1" 2>/dev/null
}

cleanup() {
    if [ -n "${BRIDGE_PID:-}" ] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill "$BRIDGE_PID" 2>/dev/null || true
        wait "$BRIDGE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
set -u

echo "[1/4] Launching Manus ROS2 -> ZMQ bridge on port ${PORT}"
if port_is_active "$PORT"; then
    if [ "$REUSE_BRIDGE" = "1" ]; then
        echo "Port ${PORT} is already active. Reusing existing ZMQ publisher because REUSE_BRIDGE=1."
    else
        ORIGINAL_PORT="$PORT"
        for candidate in $(seq "$((PORT + 1))" "$((PORT + 20))"); do
            if ! port_is_active "$candidate"; then
                PORT="$candidate"
                break
            fi
        done
        if [ "$PORT" = "$ORIGINAL_PORT" ]; then
            echo "No free ZMQ port found near ${ORIGINAL_PORT}. Set PORT manually or stop the old publisher." >&2
            exit 3
        fi
        echo "Port ${ORIGINAL_PORT} is already active. Using ${PORT} to avoid stale bridge data."
        /usr/bin/python3 geort/mocap/manus_ros2_bridge.py --glove_id "$GLOVE_ID" --port "$PORT" &
        BRIDGE_PID=$!
        sleep 2
    fi
else
    /usr/bin/python3 geort/mocap/manus_ros2_bridge.py --glove_id "$GLOVE_ID" --port "$PORT" &
    BRIDGE_PID=$!
    sleep 2
fi

echo "[2/4] Recording ${DURATION}s into data/${NAME}.npy"
/usr/bin/python3 record_manus.py --name "$NAME" --duration "$DURATION" --fps "$FPS" --port "$PORT"

echo "[3/4] Exporting offline G20 retargeting analysis"
/usr/bin/python3 dex_retargeting/manus_g20_dex_retarget.py \
  --input npy \
  --npy-path "data/${NAME}.npy" \
  --config dex_retargeting/linkerhand_g20_right_vector.yml \
  --frame-alignment g20 \
  --auto-scale \
  --output-path "analysis/${NAME}_g20_dex_retargeting.npz" \
  --print-every 0

echo "[4/4] Rendering G20 visualization"
/usr/bin/python3 dex_retargeting/visualize_manus_g20.py \
  --input-path "data/${NAME}.npy" \
  --config dex_retargeting/linkerhand_g20_right_vector.yml \
  --frame-alignment g20 \
  --output-path "analysis/${NAME}_g20_visualization.gif" \
  --frame-step 1 \
  --fps 18 \
  --auto-scale

echo
echo "Done."
echo "Recorded data:        data/${NAME}.npy"
echo "Retargeting archive:  analysis/${NAME}_g20_dex_retargeting.npz"
echo "Visualization GIF:    analysis/${NAME}_g20_visualization.gif"
echo "Visualization cover:  analysis/${NAME}_g20_visualization.png"
