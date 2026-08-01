#!/bin/bash
# Entrypoint for the Apollo SIL container.
# Sources either a local apollo build (mounted overlay) or the image's baked
# install as a fallback. Builds sim_subsystems in-container only when the
# overlay does not already provide it.

set -e

echo "========================================="
echo "  Proxie SIL Container Entrypoint"
echo "========================================="
echo ""

# 1. ROS2 environment (set before any source so nothing overrides)
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

# 2. Source the ROS base.
# The Apollo dev container builds against Vulcanexus Humble (eProsima's rosidl
# with typesupport_identifier_v2). The overlay's .so files link those symbols,
# so we must source Vulcanexus here — not stock /opt/ros/humble.
if [ -f /opt/vulcanexus/humble/setup.bash ]; then
    source /opt/vulcanexus/humble/setup.bash
    echo "✓ Sourced Vulcanexus Humble"
else
    source /opt/ros/humble/setup.bash
    echo "✓ Sourced stock ROS2 Humble (Vulcanexus not available)"
fi

# 3. Source the apollo install — overlay (local dev build) or baked image.
LOCAL_OVERLAY=/home/cobot/apollo_local_install
if [ "${APOLLO_LOCAL_OVERLAY:-0}" = "1" ] && [ -d "$LOCAL_OVERLAY" ]; then
    # Mounted from developer's .ros_container_install or install/.
    source "$LOCAL_OVERLAY/setup.bash" 2>/dev/null || \
        source "$LOCAL_OVERLAY/local_setup.bash" 2>/dev/null || true
    echo "✓ Using LOCAL apollo build (mounted overlay)"
    echo "  path: $LOCAL_OVERLAY"
    echo "  (rebuild with: ./run_container.sh -- colcon build --packages-up-to sim_subsystems)"

    # Expand .egg-link files into PYTHONPATH. colcon --symlink-install uses
    # egg-links for pure-Python packages, but they only resolve when the
    # containing site-packages is processed by site.py (which doesn't happen
    # for colcon prefix paths). Manually add each target to PYTHONPATH.
    while IFS= read -r -d '' _egglink; do
        _target=$(head -1 "$_egglink")
        [ -d "$_target" ] && export PYTHONPATH="$_target:${PYTHONPATH:-}"
    done < <(find "$LOCAL_OVERLAY" -name "*.egg-link" -print0 2>/dev/null)
else
    # Fallback: the image's baked apollo install (from the ECR image build).
    if [ -f /home/cobot/apollo/install/setup.bash ]; then
        source /home/cobot/apollo/install/setup.bash
        echo "✓ Using BAKED apollo install (image fallback — no local build detected)"
        echo "  To use your dev build, run in the apollo container:"
        echo "    colcon build --packages-up-to sim_subsystems"
    else
        echo "⚠ No apollo install found (neither overlay nor baked)"
    fi
fi

# 4. Reassert RMW after sourcing (Vulcanexus/apollo setup.bash can reset it).
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
echo ""
echo "  RMW=$RMW_IMPLEMENTATION  DOMAIN=$ROS_DOMAIN_ID"

# 5. acados runtime (cobot_local_nav_planner links libacados.so).
if [ -d /home/cobot/acados/lib ]; then
    export LD_LIBRARY_PATH="/home/cobot/acados/lib:${LD_LIBRARY_PATH:-}"
fi

# 6. Install scenario manager ROS deps if missing.
echo ""
REQUIRED_PKGS=(ros-humble-simulation-interfaces ros-humble-nav2-msgs)
MISSING_PKGS=()
for pkg in "${REQUIRED_PKGS[@]}"; do
    dpkg -s "$pkg" &>/dev/null || MISSING_PKGS+=("$pkg")
done
if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo "✓ Installing: ${MISSING_PKGS[*]}"
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq 2>&1 | tail -1 || true
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${MISSING_PKGS[@]}" 2>&1 | tail -1 || \
        echo "  Warning: apt-get install failed"
else
    echo "✓ ROS deps already installed"
fi

# 6b. Install additional Python deps for the transfuser scripts, if missing.
echo ""
if python3 -c "import skimage" &>/dev/null; then
    echo "✓ scikit-image already installed"
else
    echo "✓ Installing scikit-image"
    pip3 install --quiet scikit-image || echo "  Warning: pip install scikit-image failed"
fi

echo ""
if python3 -c "import timm" &>/dev/null; then
    echo "✓ timm already installed"
else
    echo "✓ Installing timm"
    pip3 install --quiet timm || echo "  Warning: pip install timm failed"
fi

echo ""
if python3 -c "import tensorboard" &>/dev/null; then
    echo "tensorboard already installed"
else
    echo "Installing tensorboard"
    pip3 install --quiet tensorboard || echo " Warning: tensorboard install failed"
fi

# 7. Stage cobot_params config.
echo ""
echo "✓ Configuring cobot_params..."
sudo mkdir -p /home/cobot/apollo/config
sudo chown -R cobot:cobot /home/cobot/apollo/config
if [ -d /home/cobot/apollo/config_overrides ]; then
    cp -f /home/cobot/apollo/config_overrides/* /home/cobot/apollo/config/ 2>/dev/null || true
fi
if [ -f /home/cobot/apollo/config/cobot_params_override.yaml ]; then
    sed -i 's/global_planner_type: AStar/global_planner_type: RouteGraph/' /home/cobot/apollo/config/cobot_params_override.yaml
    echo "  RouteGraph global planner"
fi
SIL_MAP_NAME="${SIL_MAP_NAME:-patrick_henry_whole}"
echo "map: $SIL_MAP_NAME" > /home/cobot/apollo/config/map.yaml
echo "  map: $SIL_MAP_NAME"

# 8. sim_subsystems: use Apollo overlay if available, otherwise build in-container.
echo ""
BUILD_WS=/home/cobot/ros2_ws_build
if [ -f "$LOCAL_OVERLAY/sim_subsystems/share/sim_subsystems/package.xml" ] &&    [ "${APOLLO_LOCAL_OVERLAY:-0}" = "1" ]; then
    echo "✓ sim_subsystems provided by Apollo overlay (skipping in-container build)"
elif [ -d /workspace/ros2_workspace/src/sim_subsystems ]; then
    mkdir -p $BUILD_WS/src
    ln -sfn /workspace/ros2_workspace/src/sim_subsystems $BUILD_WS/src/sim_subsystems
    cd $BUILD_WS
    if [ ! -f install/sim_subsystems/share/sim_subsystems/package.xml ] ||        [ /workspace/ros2_workspace/src/sim_subsystems/package.xml -nt install/sim_subsystems/share/sim_subsystems/package.xml ]; then
        echo "✓ Building sim_subsystems (launch files + helper nodes)..."
        colcon build --symlink-install --packages-select sim_subsystems 2>&1             | grep -E "Failed|Finished|Summary" || true
    else
        echo "✓ sim_subsystems already built"
    fi
    [ -f install/setup.bash ] && source install/setup.bash
    cd /workspace
fi

# 9. Final RMW reassert after workspace source.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "Commands:"
echo "  ros2 launch sim_subsystems sim.launch.py"
echo "  python3 /workspace/scripts/scenario_manager/sample_patrick_henry_ebc_smooth.py"
echo ""
echo "========================================="
echo ""

cd /workspace
exec bash --rcfile /workspace/container/setup_ros2_workspace.sh
