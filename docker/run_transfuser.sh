#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sil_ecr_auth.sh"
sil_ensure_aws_and_ecr
sil_ensure_standalone_image_local

SIL_IMAGE="${SIL_ECR_DOMAIN}/apollo-nav-standalone:latest"
TRANSFUSER_CONTAINER_NAME="${TRANSFUSER_CONTAINER_NAME:-proxie-transfuser}"

# ---------------------------------------------------------------------------
# If the container is already running, just open a new shell into it
# instead of trying to start/reuse the whole docker run pipeline.
# ---------------------------------------------------------------------------
if docker ps --format '{{.Names}}' | grep -qx "$TRANSFUSER_CONTAINER_NAME"; then
	echo "Container '$TRANSFUSER_CONTAINER_NAME' is already running — attaching a new shell."
	exec docker exec -it "$TRANSFUSER_CONTAINER_NAME" bash
fi

APOLLO_DIR="${APOLLO_DIR:-$HOME/repos/apollo}"
# sim_subsystems is no longer a hard requirement for this script.
# if [ ! -d "$APOLLO_DIR/src/sim_subsystems" ]; then
# 	echo "ERROR: \$APOLLO_DIR=$APOLLO_DIR does not contain src/sim_subsystems."
# 	echo "       Set APOLLO_DIR to the apollo checkout that has the sim_subsystems package."
# 	exit 1
# fi

# ---------------------------------------------------------------------------
# Detect the developer's apollo colcon build (--symlink-install).
# Priority: .ros_container_install (docker build) > install/ (host build).
# If found, mount it as the overlay; otherwise the baked image install is used.
# ---------------------------------------------------------------------------
APOLLO_OVERLAY_MOUNTS=()
APOLLO_OVERLAY_ENV=()
for _inst in "$APOLLO_DIR/.ros_container_install" "$APOLLO_DIR/install"; do
	[ -f "$_inst/setup.bash" ] || continue
	_bld="$(dirname "$_inst")/$( [ "$(basename "$_inst")" = ".ros_container_install" ] && echo .ros_container_build || echo build )"
	APOLLO_OVERLAY_MOUNTS=(
		-v "$_inst:/home/cobot/apollo_local_install:ro"
	)
	[ -d "$_bld" ] && APOLLO_OVERLAY_MOUNTS+=(-v "$_bld:/home/cobot/apollo/build:ro")
	[ -d "$APOLLO_DIR/src" ] && APOLLO_OVERLAY_MOUNTS+=(-v "$APOLLO_DIR/src:/home/cobot/apollo/src:ro")
	[ -d "$APOLLO_DIR/third-party" ] && APOLLO_OVERLAY_MOUNTS+=(-v "$APOLLO_DIR/third-party:/home/cobot/apollo/third-party:ro")
	APOLLO_OVERLAY_ENV=(-e APOLLO_LOCAL_OVERLAY=1)
	echo "Mounting local apollo build: $_inst"
	break
done
if [ ${#APOLLO_OVERLAY_MOUNTS[@]} -eq 0 ]; then
	echo "No local apollo build found — will use the image's baked install."
	echo "  (Build in apollo container: ./run_container.sh -- colcon build --packages-up-to sim_subsystems)"
fi

# acados runtime libs (cobot_local_nav_planner links libacados.so).
# Auto-extract from the Apollo dev image on first run.
# Disabled: requires the locally-built apollo-ros-humble-20 dev image, which isn't available here.
ACADOS_CACHE="$PROJECT_DIR/.acados"
# if [ ! -d "$ACADOS_CACHE/lib" ]; then
# 	APOLLO_DEV_IMAGE="apollo-ros-humble-20:latest"
# 	echo "Extracting acados runtime from $APOLLO_DEV_IMAGE..."
# 	mkdir -p "$ACADOS_CACHE"
# 	docker run --rm -v "$ACADOS_CACHE:/out" "$APOLLO_DEV_IMAGE" \
# 		cp -a /opt/acados/lib /opt/acados/include /out/
# 	echo "Cached at $ACADOS_CACHE"
# fi
ACADOS_MOUNT=(-v "$ACADOS_CACHE:/home/cobot/acados:ro")

# sim_subsystems is no longer a hard requirement for this script.
SIM_SUBSYSTEMS_MOUNT=()
# SIM_SUBSYSTEMS_MOUNT=(-v "$APOLLO_DIR/src/sim_subsystems:/workspace/ros2_workspace/src/sim_subsystems:rw")

# No local apollo checkout available — skip robot config overrides and the
# DDS profile mount; the image's baked-in defaults are used instead.
APOLLO_CONFIG_MOUNTS=()
# APOLLO_CONFIG_MOUNTS=(
# 	-v "$APOLLO_DIR/robot_configurations/local-development:/home/cobot/apollo/config_overrides:ro"
# 	-v "$APOLLO_DIR/scripts/dds_config/sil_dds.xml:/home/cobot/fastrtps_profile.xml:ro"
# )

xhost +local:docker 2>/dev/null || true

sil_try_reuse_named_container "$TRANSFUSER_CONTAINER_NAME" || docker run -it --rm \
	--gpus all \
	--network host \
	--ipc host \
	--name "$TRANSFUSER_CONTAINER_NAME" \
	-e DISPLAY="${DISPLAY:-:0}" \
	-e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}" \
	-e ROS_LOCALHOST_ONLY=0 \
	-e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
	-e FASTRTPS_DEFAULT_PROFILES_FILE=/home/cobot/fastrtps_profile.xml \
	-e FASTDDS_DEFAULT_PROFILES_FILE=/home/cobot/fastrtps_profile.xml \
	-e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
	-e SIL_MAP_NAME="${SIL_MAP_NAME:-patrick_henry_whole}" \
	"${APOLLO_OVERLAY_ENV[@]}" \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v "$PROJECT_DIR/container_files/transfuser:/workspace/:rw" \
	-v "$PROJECT_DIR/assets:/workspace/assets:ro" \
	"${SIM_SUBSYSTEMS_MOUNT[@]}" \
	"${APOLLO_CONFIG_MOUNTS[@]}" \
	"${ACADOS_MOUNT[@]}" \
	"${APOLLO_OVERLAY_MOUNTS[@]}" \
	"${SIL_IMAGE}" \
	/workspace/container/entrypoint_sil.sh   # ASSUMPTION: same entrypoint, adjust if transfuser has its own