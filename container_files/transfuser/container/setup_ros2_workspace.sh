#!/bin/bash
# Interactive-shell rcfile for the SIL container.
# Usage: source /workspace/container/setup_ros2_workspace.sh

# RMW first — before any source that might reset it.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

# Source Vulcanexus (matches apollo dev build ABI).
if [ -f /opt/vulcanexus/humble/setup.bash ]; then
    source /opt/vulcanexus/humble/setup.bash
else
    source /opt/ros/humble/setup.bash
fi

# Apollo install: overlay (local dev build) or baked image fallback.
LOCAL_OVERLAY=/home/cobot/apollo_local_install
if [ "${APOLLO_LOCAL_OVERLAY:-0}" = "1" ] && [ -d "$LOCAL_OVERLAY" ]; then
    source "$LOCAL_OVERLAY/setup.bash" 2>/dev/null || \
        source "$LOCAL_OVERLAY/local_setup.bash" 2>/dev/null || true
    # Expand egg-links (--symlink-install pure-Python packages).
    while IFS= read -r -d '' _egglink; do
        _target=$(head -1 "$_egglink")
        [ -d "$_target" ] && export PYTHONPATH="$_target:${PYTHONPATH:-}"
    done < <(find "$LOCAL_OVERLAY" -name "*.egg-link" -print0 2>/dev/null)
    _SIL_SOURCE="LOCAL overlay ($LOCAL_OVERLAY)"
elif [ -f /home/cobot/apollo/install/setup.bash ]; then
    source /home/cobot/apollo/install/setup.bash
    _SIL_SOURCE="BAKED image install"
fi

# sim_subsystems in-container workspace (only exists when the Apollo overlay
# does not provide sim_subsystems; see entrypoint_sil.sh step 8).
if [ -f /home/cobot/ros2_ws_build/install/setup.bash ]; then
    source /home/cobot/ros2_ws_build/install/setup.bash
fi

# Reassert after sourcing.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"

# acados runtime.
[ -d /home/cobot/acados/lib ] && export LD_LIBRARY_PATH="/home/cobot/acados/lib:${LD_LIBRARY_PATH:-}"

alias cleanup='/workspace/scripts/cleanup_sil.sh'
cd /workspace 2>/dev/null || true

echo "✓ RMW: $RMW_IMPLEMENTATION  DOMAIN: $ROS_DOMAIN_ID"
echo "✓ source: ${_SIL_SOURCE:-unknown}"
