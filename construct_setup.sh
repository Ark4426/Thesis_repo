#!/bin/bash
# =============================================================================
# construct_setup.sh  —  One-time setup on The Construct (theconstruct.ai)
#
# Two ways to run this:
#
#  A) Before cloning (bootstraps everything from scratch):
#       curl -sL https://raw.githubusercontent.com/Ark4426/Thesis_repo/main/construct_setup.sh | bash
#
#  B) After cloning into any directory:
#       bash ~/thesis_ws/construct_setup.sh
#       bash ~/Thesis_repo/construct_setup.sh
#
# Safe to re-run: existing clone is updated via git pull, workspace rebuilt.
# =============================================================================
set -e

REPO_URL="https://github.com/Ark4426/Thesis_repo.git"

# Detect workspace: the directory containing this script (when run directly),
# or $HOME/thesis_ws when piped through curl (BASH_SOURCE[0] will be "bash").
if [[ "${BASH_SOURCE[0]}" != "bash" && -f "${BASH_SOURCE[0]}" ]]; then
    WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    WORKSPACE="$HOME/thesis_ws"
fi

echo "========================================================"
echo "  Adaptive Fusion Thesis — Construct Setup"
echo "  Workspace: $WORKSPACE"
echo "========================================================"

# ── 1. apt packages ───────────────────────────────────────────────────────────
echo "[1/5] Installing ROS 2 apt packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    ros-humble-rtabmap-ros \
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-gazebo \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-image-transport \
    ros-humble-cv-bridge \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-sensor-msgs \
    python3-pip \
    tmux \
    2>/dev/null
echo "  apt done."

# ── 2. Python packages ────────────────────────────────────────────────────────
echo "[2/5] Installing Python packages..."
pip3 install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --quiet \
    "stable-baselines3[extra]>=2.8" \
    gymnasium \
    evo \
    scipy \
    matplotlib \
    opencv-python-headless \
    captum
echo "  pip done."

# ── 3. Clone / update repo ────────────────────────────────────────────────────
echo "[3/5] Setting up workspace at $WORKSPACE ..."
if [ ! -d "$WORKSPACE/.git" ]; then
    git clone "$REPO_URL" "$WORKSPACE"
    echo "  Cloned."
else
    echo "  Repo already exists — pulling latest..."
    git -C "$WORKSPACE" pull
fi

# ── 4. Build ROS 2 package ────────────────────────────────────────────────────
echo "[4/5] Building adaptive_fusion..."
source /opt/ros/humble/setup.bash
cd "$WORKSPACE"
colcon build --packages-select adaptive_fusion --symlink-install 2>&1 | tail -8
echo "  Build done."

# ── 5. Environment + directories ─────────────────────────────────────────────
echo "[5/5] Configuring environment..."
mkdir -p "$WORKSPACE/models/logs" "$WORKSPACE/models/checkpoints"

MARKER="# === thesis adaptive_fusion ==="
if ! grep -q "$MARKER" ~/.bashrc; then
    cat >> ~/.bashrc << EOF

$MARKER
source /opt/ros/humble/setup.bash
source $WORKSPACE/install/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
export GAZEBO_MODEL_PATH=$WORKSPACE/install/adaptive_fusion/share/adaptive_fusion/models:\$GAZEBO_MODEL_PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
EOF
    echo "  ~/.bashrc updated."
else
    echo "  ~/.bashrc already set — skipping."
fi

echo ""
echo "========================================================"
echo "  DONE. Run these next:"
echo ""
echo "  source ~/.bashrc"
echo ""
echo "  # Verify the stack works (~2 min):"
echo "  bash $WORKSPACE/construct_smoketest.sh"
echo ""
echo "  # Start / resume training:"
echo "  bash $WORKSPACE/construct_train.sh"
echo "========================================================"
