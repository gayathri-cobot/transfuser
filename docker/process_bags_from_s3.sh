#!/bin/bash
# Pull bag runs from S3 in windows, preprocess each window in the
# apollo-nav-standalone container, push the results to the processed bucket,
# then delete the LOCAL copies before moving to the next window.
#
#   source: s3://e2e-local-nav-data-isaac-sim-.../scenario_2/<run>/
#   dest:   s3://e2e-local-nav-processed-.../scenario_2/<run>/
#
# Only local files are ever deleted; nothing in the source bucket is modified.
#
# The runs are ~9.3 GiB each (scenario_2 is ~54 of them, ~472 GiB), so they can't
# all be on disk at once. Parallelism is therefore shaped by disk, not cores:
#
#   * a WINDOW of BATCH_RUNS runs is on disk at a time and extracted
#     concurrently (one process per run inside the container)
#   * segmentation runs ONCE per window over every frame the window produced,
#     so the SegFormer weights + CUDA context are paid for once per window
#     instead of once per run
#   * while a window is being preprocessed and pushed, the NEXT window's
#     ~9.3 GiB pulls run in the background, so S3 transfer overlaps compute
#
# BATCH_RUNS is auto-capped to what the filesystem can actually hold; see the
# disk budget below. Set BATCH_RUNS=1 PREFETCH=0 to get the old serial behaviour.
#
# Usage:
#   ./process_bags_from_s3.sh                      # every run under the prefix
#   ./process_bags_from_s3.sh <run> [<run>...]     # only these runs
#   DRY_RUN=1 ./process_bags_from_s3.sh            # show the plan, do nothing
#   FAIL_FAST=1 ./process_bags_from_s3.sh          # stop at the first failure
#   BATCH_RUNS=4 EXTRACT_JOBS=4 ./process_bags_from_s3.sh
#   MIN_DIST=0.25 ./process_bags_from_s3.sh        # metres of travel between saved frames
#   LEGACY=1 ./process_bags_from_s3.sh             # one run at a time via data_preprocess_distance.py
#
# Runs already present in the destination are skipped, so this is safe to
# re-run after an interruption.
set -uo pipefail
export CONTAINER_NAME="sil-data-preprocess-$$"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_BUCKET="${SRC_BUCKET:-e2e-local-nav-eval}"
SRC_PREFIX="${SRC_PREFIX:-scenario_3}"
DST_BUCKET="${DST_BUCKET:-e2e-local-nav-processed}"
DST_PREFIX="${DST_PREFIX:-$SRC_PREFIX}"

BAG_DATA_DIR="${BAG_DATA_DIR:-$HOME/bag_data}"
OUT_DIR="${OUT_DIR:-$HOME/data}"
ROUTES_DIR="${ROUTES_DIR:-$HOME/routes}"
LOG_DIR="${LOG_DIR:-$HOME/preprocess_logs}"
MIN_FREE_GIB="${MIN_FREE_GIB:-40}"

# Parallelism / disk budget.
#   BATCH_RUNS    runs held on disk and extracted concurrently (auto-capped)
#   EXTRACT_JOBS  extraction processes in the container (defaults to BATCH_RUNS)
#   PREFETCH      overlap the next window's S3 pull with this window's compute
#   BAG_GIB       per-run source size, for the disk budget only
#   OUT_GIB       per-run output size, for the disk budget only
BATCH_RUNS="${BATCH_RUNS:-3}"
EXTRACT_JOBS="${EXTRACT_JOBS:-$BATCH_RUNS}"
SEG_BATCH_SIZE="${SEG_BATCH_SIZE:-16}"
SEG_WORKERS="${SEG_WORKERS:-4}"
PREFETCH="${PREFETCH:-1}"
BAG_GIB="${BAG_GIB:-10}"
OUT_GIB="${OUT_GIB:-2}"
LEGACY="${LEGACY:-0}"

# Frame selection, passed through to data_preprocess_distance.save_synced_frames.
# Unset means "use the script's own defaults" (0.05 m / 0.1 s), so exporting an
# empty value here would be wrong — only forward them when actually set.
MIN_DIST="${MIN_DIST:-}"
MAX_SYNC_DT="${MAX_SYNC_DT:-}"

export AWS_CONFIG_FILE="${AWS_CONFIG_FILE:-$HOME/aws_config.ini}"
export AWS_PROFILE="${AWS_PROFILE:-sil-bag-upload}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

[ -x "${SCRIPT_DIR}/run_data_preprocess.sh" ] || die "run_data_preprocess.sh not found next to this script"
command -v aws >/dev/null || die "aws CLI not found"
aws sts get-caller-identity >/dev/null 2>&1 || die "no valid AWS session — run: aws sso login --use-device-code"

# find_command_point() in data_preprocess_distance.py hardcodes
# /workspace/routes/<scenario>.xml and is called for every run, so a missing file
# fails all 54 runs. Check once.
# [ -f "$ROUTES_DIR/${SRC_PREFIX}.xml" ] \
# 	|| die "missing $ROUTES_DIR/${SRC_PREFIX}.xml — data_preprocess_distance.py needs route definitions for '$SRC_PREFIX' (set ROUTES_DIR=...)"

# prep_extract.py / run_prep.py import data_preprocess_distance, which
# run_data_preprocess.sh bind-mounts from PREPROCESS_PY. All four have to be in
# PREP_DIR together or the container import fails per bag.
PREPROCESS_PY="${PREPROCESS_PY:-$HOME/data_preprocess_distance.py}"
[ -f "$PREPROCESS_PY" ] || die "PREPROCESS_PY not found: $PREPROCESS_PY (set PREPROCESS_PY=...)"
export PREPROCESS_PY

if [ "$LEGACY" != "1" ]; then
	PREP_DIR="${PREP_DIR:-$HOME}"
	for f in run_prep.py prep_extract.py prep_segment.py; do
		[ -f "$PREP_DIR/$f" ] || die "PREP_DIR is missing $f: $PREP_DIR/$f (set PREP_DIR=..., or LEGACY=1 to use data_preprocess_distance.py directly)"
	done
	export PREP_DIR
fi

mkdir -p "$BAG_DATA_DIR" "$OUT_DIR" "$LOG_DIR"

free_gib() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }

# ---------------------------------------------------------------------------
# Disk budget. During preprocessing of window i, peak usage is:
#   window i bags + window i+1 bags (prefetch) + window i outputs
# so cap BATCH_RUNS at what actually fits rather than trusting the env var.
# ---------------------------------------------------------------------------
if [ "$LEGACY" = "1" ]; then
	log "LEGACY=1 — one run at a time through data_preprocess_distance.py"
	BATCH_RUNS=1; PREFETCH=0; EXTRACT_JOBS=1
fi

avail_gib="$(free_gib "$BAG_DATA_DIR")"
per_run_gib=$(( BAG_GIB + OUT_GIB ))
[ "$PREFETCH" = "1" ] && per_run_gib=$(( per_run_gib + BAG_GIB ))
budget_gib=$(( avail_gib - MIN_FREE_GIB ))
max_batch=$(( budget_gib / per_run_gib ))

if [ "$max_batch" -lt 1 ]; then
	if [ "$PREFETCH" = "1" ]; then
		log "only ${avail_gib} GiB free — not enough for prefetch, disabling it"
		PREFETCH=0
		per_run_gib=$(( BAG_GIB + OUT_GIB ))
		max_batch=$(( budget_gib / per_run_gib ))
	fi
	[ "$max_batch" -lt 1 ] && die "only ${avail_gib} GiB free on $BAG_DATA_DIR; need at least $(( MIN_FREE_GIB + BAG_GIB + OUT_GIB )) GiB for a single run (lower MIN_FREE_GIB or free space)"
fi

if [ "$BATCH_RUNS" -gt "$max_batch" ]; then
	log "capping BATCH_RUNS ${BATCH_RUNS} -> ${max_batch} (${avail_gib} GiB free, ${per_run_gib} GiB needed per run in flight, keeping ${MIN_FREE_GIB} GiB headroom)"
	BATCH_RUNS="$max_batch"
	[ "$EXTRACT_JOBS" -gt "$BATCH_RUNS" ] && EXTRACT_JOBS="$BATCH_RUNS"
fi

# ---------------------------------------------------------------------------
# Which runs to process
# ---------------------------------------------------------------------------
if [ "$#" -gt 0 ]; then
	RUNS=("$@")
else
	mapfile -t RUNS < <(aws s3 ls "s3://${SRC_BUCKET}/${SRC_PREFIX}/" \
		| awk '$1 == "PRE" {sub(/\/$/, "", $2); print $2}' | sort)
fi
[ "${#RUNS[@]}" -gt 0 ] || die "no runs found under s3://${SRC_BUCKET}/${SRC_PREFIX}/"

log "${#RUNS[@]} run(s) to consider"
log "source      s3://${SRC_BUCKET}/${SRC_PREFIX}/"
log "destination s3://${DST_BUCKET}/${DST_PREFIX}/"
log "window      ${BATCH_RUNS} run(s), extract jobs ${EXTRACT_JOBS}, prefetch ${PREFETCH}, seg batch ${SEG_BATCH_SIZE}"

SUMMARY_OK=(); SUMMARY_SKIP=(); SUMMARY_FAIL=()

# ---------------------------------------------------------------------------
# Resume filter up front: windows have to be built from the pending runs, so the
# "already in the destination" check can't happen mid-loop any more.
# ---------------------------------------------------------------------------
PENDING=()
for run in "${RUNS[@]}"; do
	if [ -n "$(aws s3 ls "s3://${DST_BUCKET}/${DST_PREFIX}/${run}/" 2>/dev/null | head -1)" ]; then
		SUMMARY_SKIP+=("$run")
	else
		PENDING+=("$run")
	fi
done
[ "${#SUMMARY_SKIP[@]}" -eq 0 ] || log "${#SUMMARY_SKIP[@]} run(s) already present at the destination — skipping"

if [ "${#PENDING[@]}" -eq 0 ]; then
	log "nothing to do"
	exit 0
fi
log "${#PENDING[@]} run(s) pending"

# ---------------------------------------------------------------------------
# Background pulls. Only one window is ever in flight, so a pair of parallel
# arrays is enough state.
# ---------------------------------------------------------------------------
PULL_RUNS=(); PULL_PIDS=(); PULLED_OK=()

start_pulls() {
	PULL_RUNS=(); PULL_PIDS=()
	[ "$#" -gt 0 ] || return 0
	log "pulling ${#@} run(s): $*"
	local run
	for run in "$@"; do
		mkdir -p "${BAG_DATA_DIR}/${SRC_PREFIX}/${run}"
		aws s3 sync "s3://${SRC_BUCKET}/${SRC_PREFIX}/${run}/" \
			"${BAG_DATA_DIR}/${SRC_PREFIX}/${run}/" --only-show-errors \
			>"${LOG_DIR}/pull-${run}.log" 2>&1 &
		PULL_RUNS+=("$run"); PULL_PIDS+=("$!")
	done
}

wait_pulls() {
	PULLED_OK=()
	local i run pid
	for i in "${!PULL_RUNS[@]}"; do
		run="${PULL_RUNS[$i]}"; pid="${PULL_PIDS[$i]}"
		if wait "$pid"; then
			PULLED_OK+=("$run")
		else
			log "FAILED to pull ${run} — see ${LOG_DIR}/pull-${run}.log"
			rm -rf "${BAG_DATA_DIR}/${SRC_PREFIX}/${run}"
			SUMMARY_FAIL+=("${run}:pull")
		fi
	done
	PULL_RUNS=(); PULL_PIDS=()
}

# A prefetch in flight when we bail out would otherwise leave a half-synced bag
# on disk and an orphaned aws process behind.
cleanup_pulls() {
	[ "${#PULL_PIDS[@]}" -eq 0 ] && return 0
	log "cancelling ${#PULL_PIDS[@]} in-flight pull(s)"
	local i
	for i in "${!PULL_PIDS[@]}"; do
		kill "${PULL_PIDS[$i]}" 2>/dev/null
		wait "${PULL_PIDS[$i]}" 2>/dev/null
		rm -rf "${BAG_DATA_DIR}/${SRC_PREFIX}/${PULL_RUNS[$i]}"
	done
	PULL_RUNS=(); PULL_PIDS=()
}
trap cleanup_pulls EXIT INT TERM

# ---------------------------------------------------------------------------
# Preprocess one window, then push and delete per run.
# ---------------------------------------------------------------------------
WINDOW_FAILED=0

process_window() {
	local -a window=("$@")
	local run rel winlog
	local -a rels=()
	WINDOW_FAILED=0

	for run in "${window[@]}"; do
		rels+=("${SRC_PREFIX}/${run}")
		rm -rf "${OUT_DIR}/${SRC_PREFIX}/${run}"
	done

	if [ "$LEGACY" = "1" ]; then
		winlog="${LOG_DIR}/${window[0]}.log"
	else
		winlog="${LOG_DIR}/window-${window[0]}.log"
	fi

	log "preprocessing ${#window[@]} run(s) (log: $winlog)"
	local rc=0
	if [ "$LEGACY" = "1" ]; then
		# Single mode forwards trailing args straight to
		# data_preprocess_distance.py rather than reading MIN_DIST/MAX_SYNC_DT.
		local -a legacy_args=()
		[ -n "$MIN_DIST" ] && legacy_args+=(--min-dist "$MIN_DIST")
		[ -n "$MAX_SYNC_DT" ] && legacy_args+=(--max-sync-dt "$MAX_SYNC_DT")
		BAG_DATA_DIR="$BAG_DATA_DIR" OUT_DIR="$OUT_DIR" ROUTES_DIR="$ROUTES_DIR" \
			"${SCRIPT_DIR}/run_data_preprocess.sh" "${rels[0]}" \
			${legacy_args[@]+"${legacy_args[@]}"} >"$winlog" 2>&1 || rc=$?
	else
		BAG_DATA_DIR="$BAG_DATA_DIR" OUT_DIR="$OUT_DIR" ROUTES_DIR="$ROUTES_DIR" \
		PREP_MODE=batch PREP_DIR="$PREP_DIR" PREP_LOG_DIR="${LOG_DIR}/prep" \
		EXTRACT_JOBS="$EXTRACT_JOBS" SEG_BATCH_SIZE="$SEG_BATCH_SIZE" SEG_WORKERS="$SEG_WORKERS" \
		MIN_DIST="$MIN_DIST" MAX_SYNC_DT="$MAX_SYNC_DT" \
			"${SCRIPT_DIR}/run_data_preprocess.sh" "${rels[@]}" >"$winlog" 2>&1 || rc=$?
	fi

	# run_prep.py exits non-zero if ANY run in the window failed, so per-run
	# status comes from the output tree below rather than from $rc. Log the tail
	# either way, then free the bags — they are dead weight once extracted.
	if [ "$rc" -ne 0 ]; then
		log "preprocess exited ${rc} — see $winlog (tail below)"
		tail -n 15 "$winlog" | sed 's/^/    /'
	else
		log "preprocess ok"
	fi
	for run in "${window[@]}"; do
		rm -rf "${BAG_DATA_DIR}/${SRC_PREFIX}/${run}"
	done

	# --- push, in parallel, only the runs that produced output ------------
	local -a push_runs=() push_pids=()
	for run in "${window[@]}"; do
		local local_out="${OUT_DIR}/${SRC_PREFIX}/${run}"
		if [ -z "$(ls -A "$local_out" 2>/dev/null)" ]; then
			log "FAILED ${run}: preprocess produced no output in $local_out"
			rm -rf "$local_out"
			SUMMARY_FAIL+=("${run}:empty-output")
			WINDOW_FAILED=1
			continue
		fi
		aws s3 sync "${local_out}/" "s3://${DST_BUCKET}/${DST_PREFIX}/${run}/" \
			--only-show-errors >"${LOG_DIR}/push-${run}.log" 2>&1 &
		push_runs+=("$run"); push_pids+=("$!")
	done

	local i
	for i in ${push_runs[@]+"${!push_runs[@]}"}; do
		run="${push_runs[$i]}"
		if wait "${push_pids[$i]}"; then
			log "push ok — deleting local output for ${run}"
			rm -rf "${OUT_DIR}/${SRC_PREFIX}/${run}"
			SUMMARY_OK+=("$run")
		else
			# Keep the output so a re-run can retry the push without re-pulling.
			log "FAILED to push ${run} — keeping local output at ${OUT_DIR}/${SRC_PREFIX}/${run}"
			SUMMARY_FAIL+=("${run}:push")
			WINDOW_FAILED=1
		fi
	done

	log "completed ${#SUMMARY_OK[@]}/${#PENDING[@]} this session"
}

# ---------------------------------------------------------------------------
# Main loop: pull window i+1 while window i is being preprocessed.
# ---------------------------------------------------------------------------
total="${#PENDING[@]}"

if [ "${DRY_RUN:-0}" = "1" ]; then
	i=0
	while [ "$i" -lt "$total" ]; do
		log "DRY_RUN window $(( i / BATCH_RUNS + 1 )): ${PENDING[*]:i:BATCH_RUNS}"
		i=$(( i + BATCH_RUNS ))
	done
	log "DRY_RUN: would pull each window, preprocess it in one container "\
"(extract x${EXTRACT_JOBS} then one batched segmentation pass), push, and delete local copies"
	SUMMARY_OK=("${PENDING[@]}")
	log "===== summary (dry run) ====="
	log "would process: ${#SUMMARY_OK[@]}   skipped: ${#SUMMARY_SKIP[@]}"
	trap - EXIT INT TERM
	exit 0
fi

i=0
start_pulls "${PENDING[@]:0:BATCH_RUNS}"

while [ "$i" -lt "$total" ]; do
	echo
	window=("${PENDING[@]:i:BATCH_RUNS}")
	log "=== window $(( i / BATCH_RUNS + 1 )): ${window[*]} ==="

	wait_pulls
	ready=(${PULLED_OK[@]+"${PULLED_OK[@]}"})

	next=$(( i + BATCH_RUNS ))

	# Kick off the next window's transfer now so it overlaps the container run.
	if [ "$next" -lt "$total" ]; then
		avail="$(free_gib "$BAG_DATA_DIR")"
		if [ "$avail" -lt "$MIN_FREE_GIB" ]; then
			log "only ${avail} GiB free (need ${MIN_FREE_GIB}) — not starting the next window"
			SUMMARY_FAIL+=("${PENDING[$next]}:disk")
			total="$next"          # finish the current window, then stop
		elif [ "$PREFETCH" = "1" ]; then
			start_pulls "${PENDING[@]:next:BATCH_RUNS}"
		fi
	fi

	if [ "${#ready[@]}" -gt 0 ]; then
		process_window "${ready[@]}"
		if [ "$WINDOW_FAILED" = "1" ] && [ "${FAIL_FAST:-0}" = "1" ]; then
			log "FAIL_FAST=1 — stopping"
			break
		fi
	else
		log "no runs in this window survived the pull — skipping"
		[ "${FAIL_FAST:-0}" = "1" ] && break
	fi

	i="$next"

	# Without prefetch, the pull happens here instead: serial, one window behind.
	if [ "$PREFETCH" != "1" ] && [ "$i" -lt "$total" ]; then
		start_pulls "${PENDING[@]:i:BATCH_RUNS}"
	fi
done

cleanup_pulls
trap - EXIT INT TERM

echo
log "===== summary ====="
log "ok:      ${#SUMMARY_OK[@]}"
log "skipped: ${#SUMMARY_SKIP[@]}"
log "failed:  ${#SUMMARY_FAIL[@]}   ${SUMMARY_FAIL[*]:-}"
[ "${#SUMMARY_FAIL[@]}" -eq 0 ] || exit 1
