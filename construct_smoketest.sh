#!/bin/bash
# =============================================================================
# construct_smoketest.sh  —  Quick stack validation on The Construct
#
# Launches W1 headless, waits for /fusion/fused_pose, reports PASS or FAIL.
# Run after construct_setup.sh before committing to an 8-hour training session.
#
# Usage:  bash construct_smoketest.sh
# =============================================================================

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
source "$WORKSPACE/install/setup.bash"
export TURTLEBOT3_MODEL=waffle_pi
export GAZEBO_MODEL_PATH="$WORKSPACE/install/adaptive_fusion/share/adaptive_fusion/models:$GAZEBO_MODEL_PATH"
export CUDA_VISIBLE_DEVICES=""

echo "=== Smoke test: launching W1 headless ==="

ros2 launch adaptive_fusion simulation.launch.py \
    world:=w1_static headless:=true rviz:=false \
    > /tmp/smoketest_launch.log 2>&1 &
LAUNCH_PID=$!

echo "Waiting for /fusion/fused_pose (up to 90 s)..."
TIMEOUT=90
START=$(date +%s)
GOT_POSE=0
while [ $(( $(date +%s) - START )) -lt $TIMEOUT ]; do
    if ros2 topic echo /fusion/fused_pose --once --timeout 3 2>/dev/null | grep -q "position"; then
        GOT_POSE=1
        break
    fi
    printf "."
    sleep 3
done
echo ""

# Clean up
kill $LAUNCH_PID 2>/dev/null
sleep 2
pkill -9 gzserver 2>/dev/null || true
pkill -9 gzclient 2>/dev/null || true

if [ $GOT_POSE -eq 1 ]; then
    echo ""
    echo "=== PASS — fused pose received, stack is working ==="
    echo "  Ready to train:  bash $WORKSPACE/construct_train.sh"
else
    echo ""
    echo "=== FAIL — no pose received within ${TIMEOUT}s ==="
    echo ""
    echo "  Diagnose: cat /tmp/smoketest_launch.log | tail -50"
    echo ""
    echo "  Common fixes:"
    echo "    Missing rtabmap : sudo apt install ros-humble-rtabmap-ros"
    echo "    Missing slam_tb  : sudo apt install ros-humble-slam-toolbox"
    echo "    Model path issue : check GAZEBO_MODEL_PATH includes $WORKSPACE/install/..."
    echo "    Build stale      : cd $WORKSPACE && colcon build --packages-select adaptive_fusion"
    exit 1
fi
