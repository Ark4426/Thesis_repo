#!/bin/bash
# =============================================================================
# construct_train.sh  —  Daily training launcher on The Construct
#
# Runs inside tmux so training survives browser/VNC disconnect.
#
# Usage:
#   bash construct_train.sh                          # auto-detect checkpoint
#   bash construct_train.sh path/to/checkpoint.zip  # explicit resume
#
# While training:
#   tmux attach -t thesis        # reattach to see live output
#   Ctrl-B then D                # detach (training keeps going)
#   tail -f models/logs/train_stdout.log   # watch progress from any terminal
# =============================================================================

# Self-locate workspace (works whether called as ./script or bash path/script)
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_DIR="$WORKSPACE/models"
SESSION="thesis"
RESUME="${1:-}"

source /opt/ros/humble/setup.bash
source "$WORKSPACE/install/setup.bash"
export TURTLEBOT3_MODEL=waffle_pi
export GAZEBO_MODEL_PATH="$WORKSPACE/install/adaptive_fusion/share/adaptive_fusion/models:$GAZEBO_MODEL_PATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

# ── Auto-detect latest checkpoint if not provided ─────────────────────────────
if [ -z "$RESUME" ]; then
    LATEST=$(ls -t "$SAVE_DIR/checkpoints/"*.zip 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "Latest checkpoint found: $LATEST"
        read -rp "Resume from it? [Y/n] " REPLY
        if [[ ! "$REPLY" =~ ^[Nn]$ ]]; then
            RESUME="$LATEST"
        fi
    fi
fi

# ── Guard: don't double-launch ────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' is already running."
    echo "  Attach : tmux attach -t $SESSION"
    echo "  Kill   : tmux kill-session -t $SESSION  (then re-run this script)"
    exit 1
fi

# ── Build training command ────────────────────────────────────────────────────
mkdir -p "$SAVE_DIR/logs"

TRAIN_ARGS="--total-steps 800000 \
    --save-dir $SAVE_DIR \
    --save-freq 10000 \
    --seed 0 \
    --worlds w1_static w2_low_dynamic w3_high_dynamic w4_visually_degraded"

[ -n "$RESUME" ] && TRAIN_ARGS="$TRAIN_ARGS --resume-from $RESUME"

TRAIN_CMD="source /opt/ros/humble/setup.bash && \
source $WORKSPACE/install/setup.bash && \
export TURTLEBOT3_MODEL=waffle_pi && \
export GAZEBO_MODEL_PATH=$WORKSPACE/install/adaptive_fusion/share/adaptive_fusion/models:\$GAZEBO_MODEL_PATH && \
export PYTHONUNBUFFERED=1 && \
export CUDA_VISIBLE_DEVICES='' && \
cd $WORKSPACE && \
python3 -u src/adaptive_fusion/scripts/train_ppo.py $TRAIN_ARGS \
    2>&1 | tee $SAVE_DIR/logs/train_stdout.log"

# ── Launch in tmux ────────────────────────────────────────────────────────────
echo "========================================================"
echo "  Launching training in tmux session: $SESSION"
echo "  Steps    : 800,000  |  Checkpoint every ~27 min"
[ -n "$RESUME" ] && echo "  Resuming : $RESUME" || echo "  Starting : fresh run"
echo ""
echo "  Reattach : tmux attach -t $SESSION"
echo "  Detach   : Ctrl-B then D"
echo "  Progress : tail -f $SAVE_DIR/logs/train_stdout.log"
echo "========================================================"

tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "$TRAIN_CMD" Enter

sleep 3
echo ""
echo "=== Live log (Ctrl-C stops watching, training keeps running) ==="
tail -f "$SAVE_DIR/logs/train_stdout.log"
