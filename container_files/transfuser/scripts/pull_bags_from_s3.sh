#!/usr/bin/env bash
#
# pull_bags_from_s3.sh — download rosbag runs from S3 to the local bag_data dir.
#
# Mirrors push_bags_to_s3.sh but in reverse: syncs one run prefix (or a whole
# date subtree) from S3 down to container_files/sil/bag_data, preserving the
# <date>/<subfolder>/<run> layout so the SIL container can find the bags at
# /workspace/bag_data/<date>/...
#
# `aws s3 sync` is idempotent — rerunning only fetches what's missing or changed.
#
# Usage:
#   # Download a single run:
#   ./pull_bags_from_s3.sh --bucket e2e-local-nav-data-isaac-sim-938145530947-us-west-2-an \
#       --key 2026-07-12/corridor1_w1_route1_030808
#
#   # Download everything under a date:
#   ./pull_bags_from_s3.sh --bucket e2e-local-nav-data-isaac-sim-938145530947-us-west-2-an \
#       --key 2026-07-12
#
#   # Download a whole bucket (no --key):
#   ./pull_bags_from_s3.sh --bucket e2e-local-nav-data-isaac-sim-938145530947-us-west-2-an
#
# Config precedence (first non-empty wins):
#   bucket : --bucket  -> $SIL_BAG_S3_BUCKET   (required)
#   key    : --key     -> $SIL_BAG_S3_KEY       (optional S3 prefix to narrow the sync)
#   dest   : --dest    -> $BAG_DATA_DIR         (default <repo>/container_files/sil/bag_data)
#   profile: --profile -> $AWS_PROFILE          (default: sil-bag-upload)
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/docker"

BUCKET="${SIL_BAG_S3_BUCKET:-}"
KEY="${SIL_BAG_S3_KEY:-}"
DEST="${BAG_DATA_DIR:-${REPO_ROOT}/container_files/transfuser/bag_data}"
PROFILE="${AWS_PROFILE:-sil-bag-upload}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)  BUCKET="$2"; shift 2 ;;
    --key)     KEY="$2"; shift 2 ;;
    --dest)    DEST="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --region)  REGION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

KEY="${KEY#/}"; KEY="${KEY%/}"   # strip leading/trailing slashes

# --- preflight ------------------------------------------------------------
if [[ -z "$BUCKET" ]]; then
  echo "ERROR: no bucket. Pass --bucket <name> or set SIL_BAG_S3_BUCKET." >&2
  exit 2
fi
if ! command -v aws &>/dev/null; then
  echo "ERROR: aws CLI not found on PATH." >&2
  exit 2
fi

mkdir -p "$DEST"

if [[ -f "${DOCKER_DIR}/aws_config.ini" ]]; then
  export AWS_CONFIG_FILE="${DOCKER_DIR}/aws_config.ini"
fi
[[ -n "$PROFILE" ]] && export AWS_PROFILE="$PROFILE"

if ! aws sts get-caller-identity &>/dev/null; then
  echo "Not logged in to AWS (profile=${AWS_PROFILE:-default}). Run:" >&2
  echo "    aws sso login${AWS_PROFILE:+ --profile $AWS_PROFILE}" >&2
  exit 1
fi

# --- build source / dest paths --------------------------------------------
SRC="s3://${BUCKET}${KEY:+/$KEY}/"
LOCAL_DEST="${DEST}${KEY:+/$KEY}"

echo "======================================================================"
echo "source : $SRC"
echo "dest   : $LOCAL_DEST"
echo "profile: ${AWS_PROFILE:-<default>}   region: ${REGION}"
[[ $DRY_RUN -eq 1 ]] && echo "(DRY RUN)"
echo "======================================================================"

mkdir -p "$LOCAL_DEST"

cmd=(aws s3 sync "$SRC" "$LOCAL_DEST" --region "$REGION" --no-progress)
[[ -n "${AWS_PROFILE:-}" ]] && cmd+=(--profile "$AWS_PROFILE")
[[ $DRY_RUN -eq 1 ]] && cmd+=(--dryrun)

echo "$ ${cmd[*]}"
"${cmd[@]}"

echo "======================================================================"
echo "Done. Files written to: $LOCAL_DEST"
