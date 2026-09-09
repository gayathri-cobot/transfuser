#!/bin/bash
# Run data_preprocess_distance.py inside the apollo-nav-standalone image.
#
# The image already ships everything the script imports (ROS 2 humble python
# bindings, cv2 + cv_bridge, torch, transformers, PIL), so nothing is built or
# installed here and no local apollo checkout is used.
#
# Two modes:
#
#   PREP_MODE=single (default) — one run through data_preprocess_distance.py,
#                                serially.
#   PREP_MODE=batch            — N runs through run_prep.py: extraction fanned
#                                out across processes, then ONE batched
#                                segmentation pass over all of them. The model
#                                is loaded once per invocation instead of once
#                                per run, so pass as many runs as fit on disk.
#
# Usage:
#   ./run_data_preprocess.sh <run-dir> [extra data_preprocess_distance.py args...]
#   ./run_data_preprocess.sh scenario_2/corridor1_w1_route1_004416
#   ./run_data_preprocess.sh scenario_2/corridor1_w1_route1_004416 --max-sync-dt 2.0
#   PREP_MODE=batch EXTRACT_JOBS=4 ./run_data_preprocess.sh scenario_2/run_a scenario_2/run_b
#   SHELL_ONLY=1 ./run_data_preprocess.sh          # drop into a shell instead
#
# <run-dir> is a path relative to BAG_DATA_DIR and must contain metadata.yaml.
#
# Host layout (override any of these with env vars):
#   BAG_DATA_DIR   $HOME/bag_data   -> /workspace/bag_data   (ro)   bag runs
#   ROUTES_DIR     $HOME/routes     -> /workspace/routes     (ro)   <scenario>.xml
#   OUT_DIR        $HOME/data       -> /workspace/data       (rw)   results
#   CACHE_DIR      $HOME/.cache/sil-preprocess                      HF model cache
#   PREPROCESS_PY  $HOME/data_preprocess_distance.py
#                                   -> /workspace/data_preprocess_distance.py (ro)
#                                   prep_extract.py / run_prep.py import it by
#                                   that name, so the basename matters.
#
# PREP_MODE=batch additionally mounts (see PREP_DIR / PREP_LOG_DIR below):
#   PREP_DIR       $HOME            -> run_prep.py, prep_extract.py, prep_segment.py
#   PREP_LOG_DIR   $HOME/preprocess_logs/prep -> /workspace/prep_logs (rw)
#                                       per-bag stage 1 logs, kept on the host
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BAG_DATA_DIR="${BAG_DATA_DIR:-$HOME/bag_data}"
ROUTES_DIR="${ROUTES_DIR:-$HOME/routes}"
OUT_DIR="${OUT_DIR:-$HOME/data}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/sil-preprocess}"
PREPROCESS_PY="${PREPROCESS_PY:-$HOME/data_preprocess_distance.py}"
CONTAINER_NAME="${CONTAINER_NAME:-sil-data-preprocess}"

# PREP_MODE=batch only
PREP_MODE="${PREP_MODE:-single}"
PREP_DIR="${PREP_DIR:-$HOME}"
PREP_LOG_DIR="${PREP_LOG_DIR:-$HOME/preprocess_logs/prep}"
EXTRACT_JOBS="${EXTRACT_JOBS:-4}"
SEG_BATCH_SIZE="${SEG_BATCH_SIZE:-16}"
SEG_WORKERS="${SEG_WORKERS:-4}"

SIL_ECR_DOMAIN="${SIL_ECR_DOMAIN:-458214780330.dkr.ecr.us-west-2.amazonaws.com}"
SIL_IMAGE="${SIL_IMAGE:-${SIL_ECR_DOMAIN}/apollo-nav-standalone:latest}"

die() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Resolve how to talk to Docker before anything else. A shell that isn't in the
# docker group yet (freshly added, not re-logged-in) makes every docker call
# fail; checking here keeps that from being misreported as "image missing".
# ---------------------------------------------------------------------------
DOCKER="${DOCKER:-docker}"
if ! $DOCKER info &>/dev/null; then
	if sudo -n docker info &>/dev/null; then
		DOCKER="sudo docker"
		echo "note: this shell can't reach the Docker daemon directly — using 'sudo docker'." >&2
		echo "      (log out and back in to pick up your docker group membership)" >&2
	else
		die "cannot talk to the Docker daemon. Is it running, and are you in the docker group? (log out/in after usermod, or set DOCKER='sudo docker')"
	fi
fi

# ---------------------------------------------------------------------------
# Only touch AWS/ECR if the image isn't already here. Pulling 21GB needs a
# valid SSO session; a local image needs nothing.
# ---------------------------------------------------------------------------
if ! $DOCKER image inspect "$SIL_IMAGE" &>/dev/null; then
	echo "Image $SIL_IMAGE not present locally — authenticating to ECR to pull it."
	# shellcheck source=/dev/null
	source "${SCRIPT_DIR}/sil_ecr_auth.sh"
	sil_ensure_aws_and_ecr
	sil_ensure_standalone_image_local
fi

# ---------------------------------------------------------------------------
# The image installs transformers into /home/cobot/.local (pip --user) and ROS
# is not sourced by /etc, only by cobot's dotfiles. A direct `docker run ...
# python3` therefore fails on `import rosbag2_py`. Source both setup files
# explicitly and leave HOME=/home/cobot alone so python's user-site resolves.
# umask 002 makes results group-writable for the host user (see --user below).
# ---------------------------------------------------------------------------
CONTAINER_SETUP='umask 002
source /opt/ros/humble/setup.bash
source /home/cobot/apollo/install/setup.bash 2>/dev/null || true'

# Extra bind mounts, populated by PREP_MODE=batch.
EXTRA_MOUNTS=()

# Shared validation for one run dir relative to BAG_DATA_DIR. Docker silently
# creates missing bind mount sources as empty root-owned dirs, which turns a typo
# into a confusing in-container failure, so check before we run anything.
validate_run() {
	local rel="$1" scenario
	[ -d "$BAG_DATA_DIR/$rel" ] || die "run dir not found: $BAG_DATA_DIR/$rel"
	[ -f "$BAG_DATA_DIR/$rel/metadata.yaml" ] || die "$BAG_DATA_DIR/$rel has no metadata.yaml — not a rosbag2 run dir"

	# find_command_point() hardcodes /workspace/routes/<scenario>.xml, where
	# <scenario> is the parent dir of the run (falling back to scenario_1).
	# It now falls back to synthesising waypoints from tf when that file is
	# missing, but a present route file is still the accurate path, so keep
	# failing loudly here rather than silently producing generated waypoints.
	scenario="$(basename "$(dirname "$rel")")"
	case "$scenario" in *scenario*) ;; *) scenario="scenario_1" ;; esac
	[ -d "$ROUTES_DIR" ] || die "ROUTES_DIR not found: $ROUTES_DIR (data_preprocess_distance.py needs /workspace/routes/${scenario}.xml)"
	[ -f "$ROUTES_DIR/${scenario}.xml" ] || die "missing route definitions: $ROUTES_DIR/${scenario}.xml"
}

if [ "${SHELL_ONLY:-0}" = "1" ]; then
	INNER_CMD='exec bash'
elif [ "$PREP_MODE" = "batch" ]; then
	[ "$#" -ge 1 ] || die "usage: PREP_MODE=batch $(basename "$0") <run-dir> [<run-dir>...]   (relative to $BAG_DATA_DIR)"

	[ -f "$PREPROCESS_PY" ] || die "PREPROCESS_PY not found: $PREPROCESS_PY"
	[ -d "$BAG_DATA_DIR" ] || die "BAG_DATA_DIR not found: $BAG_DATA_DIR (set BAG_DATA_DIR=... to point at your bag runs)"
	for f in run_prep.py prep_extract.py prep_segment.py; do
		[ -f "$PREP_DIR/$f" ] || die "PREP_DIR is missing $f: $PREP_DIR/$f (set PREP_DIR=...)"
	done

	CONTAINER_RUNS=()
	for rel in "$@"; do
		rel="${rel#/}"; rel="${rel%/}"
		validate_run "$rel"
		CONTAINER_RUNS+=("/workspace/bag_data/$rel")
	done

	mkdir -p "$PREP_LOG_DIR"
	EXTRA_MOUNTS=(
		-v "$PREP_DIR/run_prep.py:/workspace/run_prep.py:ro"
		-v "$PREP_DIR/prep_extract.py:/workspace/prep_extract.py:ro"
		-v "$PREP_DIR/prep_segment.py:/workspace/prep_segment.py:ro"
		-v "$PREP_LOG_DIR:/workspace/prep_logs:rw"
	)

	# --log-dir lands on the host so a failed bag stays debuggable after the
	# container is gone. No --resume: the S3 driver owns resume decisions, and a
	# done-marker would wrongly skip a run whose local bag was already deleted.
	PREP_ARGS=(--jobs "$EXTRACT_JOBS"
	           --batch-size "$SEG_BATCH_SIZE"
	           --num-workers "$SEG_WORKERS"
	           --log-dir /workspace/prep_logs)
	[ -n "${MAX_SYNC_DT:-}" ] && PREP_ARGS+=(--max-sync-dt "$MAX_SYNC_DT")
	[ -n "${MIN_DIST:-}" ] && PREP_ARGS+=(--min-dist "$MIN_DIST")

	INNER_CMD="exec python3 /workspace/run_prep.py $(printf '%q ' "${CONTAINER_RUNS[@]}" "${PREP_ARGS[@]}")"
else
	[ "$#" -ge 1 ] || die "usage: $(basename "$0") <run-dir> [extra args...]   (run-dir is relative to $BAG_DATA_DIR)"
	RUN_REL="${1#/}"; RUN_REL="${RUN_REL%/}"; shift

	[ -f "$PREPROCESS_PY" ] || die "PREPROCESS_PY not found: $PREPROCESS_PY"
	[ -d "$BAG_DATA_DIR" ] || die "BAG_DATA_DIR not found: $BAG_DATA_DIR (set BAG_DATA_DIR=... to point at your bag runs)"
	validate_run "$RUN_REL"

	INNER_CMD="exec python3 /workspace/data_preprocess_distance.py $(printf '%q ' "/workspace/bag_data/$RUN_REL" "$@")"
fi
CONTAINER_CMD=(bash -c "${CONTAINER_SETUP}
${INNER_CMD}")

# Create every bind-mount source as the invoking user. Docker would otherwise
# create missing sources itself, as root-owned empty dirs.
mkdir -p "$OUT_DIR" "$CACHE_DIR" "$BAG_DATA_DIR" "$ROUTES_DIR"

# -t only when we actually have a terminal, so this stays usable from cron/CI.
TTY_ARGS=(-i)
[ -t 0 ] && [ -t 1 ] && TTY_ARGS=(-i -t)

# ---------------------------------------------------------------------------
# GPU. `--gpus all` needs the NVIDIA Container Toolkit on the host, not just
# the driver; without it Docker 29 fails with
#   failed to discover GPU vendor from CDI: no known GPU vendor found
# SegFormer falls back to CPU (slow but correct) when no GPU is passed through.
# ---------------------------------------------------------------------------
gpu_supported() {
	command -v nvidia-container-runtime &>/dev/null && return 0
	command -v nvidia-ctk &>/dev/null && return 0
	$DOCKER info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia && return 0
	compgen -G "/etc/cdi/*.json" >/dev/null 2>&1 && return 0
	compgen -G "/var/run/cdi/*.json" >/dev/null 2>&1 && return 0
	return 1
}

GPU_ARGS=()
if [ "${NO_GPU:-0}" = "1" ]; then
	echo "NO_GPU=1 — running on CPU."
elif gpu_supported; then
	GPU_ARGS=(--gpus all)
else
	echo "WARNING: NVIDIA Container Toolkit not installed — running on CPU (torch.cuda.is_available() will be False)." >&2
	echo "         Install it to use the GPU:" >&2
	echo "           sudo apt-get install -y nvidia-container-toolkit \\" >&2
	echo "             && sudo nvidia-ctk runtime configure --runtime=docker \\" >&2
	echo "             && sudo systemctl restart docker" >&2
fi

# Run as uid 1002 (the image's cobot) so /home/cobot — mode 750, and home to
# the pip --user packages — stays readable, but with the *host* gid so anything
# written to OUT_DIR is group-owned by you and (with umask 002) group-writable.
# Running as $(id -u) instead breaks `import transformers`.
exec $DOCKER run --rm "${TTY_ARGS[@]}" \
	"${GPU_ARGS[@]}" \
	--name "$CONTAINER_NAME" \
	--user "1002:$(id -g)" \
	--network host \
	--ipc host \
	-e HF_HOME=/workspace/.hf_cache \
	-e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}" \
	-e ROS_LOCALHOST_ONLY=1 \
	-e PYTHONUNBUFFERED=1 \
	-v "$PREPROCESS_PY:/workspace/data_preprocess_distance.py:ro" \
	-v "$BAG_DATA_DIR:/workspace/bag_data:ro" \
	-v "$ROUTES_DIR:/workspace/routes:ro" \
	-v "$OUT_DIR:/workspace/data:rw" \
	-v "$CACHE_DIR:/workspace/.hf_cache:rw" \
	${EXTRA_MOUNTS[@]+"${EXTRA_MOUNTS[@]}"} \
	"$SIL_IMAGE" \
	"${CONTAINER_CMD[@]}"
