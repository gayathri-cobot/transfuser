#!/bin/bash
# Pulls scenario_1_2026-07-29/ from the e2e-local-nav-processed S3 bucket into data/.
# Meant to run inside the transfuser container (cwd /workspace). The container
# has no AWS CLI baked in and no persisted credentials (it runs with --rm), so
# both are set up fresh here on every run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUCKET="e2e-local-nav-processed-938145530947-us-west-2-an"
PREFIX="scenario_1_2026-07-29"
DEST="${SCRIPT_DIR}/data/${PREFIX}"
JOBS="${JOBS:-20}"
AWS_PROFILE_NAME="sil-bag-upload"

if ! command -v aws &>/dev/null; then
	echo "AWS CLI not found; installing AWS CLI v2..."
	_tmp="$(mktemp -d)"
	pushd "${_tmp}" >/dev/null
	SYSTEM_ARCH="$(uname -m)"
	curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${SYSTEM_ARCH}.zip" -o awscliv2.zip
	if ! command -v unzip &>/dev/null; then
		sudo apt-get update -qq
		sudo apt-get install -y -qq unzip
	fi
	unzip -q -o awscliv2.zip
	sudo ./aws/install
	popd >/dev/null
	rm -rf "${_tmp}"
fi

if ! command -v aws &>/dev/null; then
	echo "error: AWS CLI still not available after install attempt." >&2
	exit 1
fi

mkdir -p "${HOME}/.aws"
if ! grep -q "profile ${AWS_PROFILE_NAME}" "${HOME}/.aws/config" 2>/dev/null; then
	cat >>"${HOME}/.aws/config" <<-EOF

	[sso-session cobot-aws-sso]
	sso_start_url = https://cobot.awsapps.com/start#
	sso_region = us-west-2

	[profile ${AWS_PROFILE_NAME}]
	sso_session = cobot-aws-sso
	sso_account_id = 938145530947
	sso_role_name = AdministratorAccess
	region = us-west-2
	output = json
	EOF
fi
export AWS_PROFILE="${AWS_PROFILE:-${AWS_PROFILE_NAME}}"

if ! aws sts get-caller-identity &>/dev/null; then
	echo "Not logged in to AWS profile '${AWS_PROFILE}'. Starting SSO device-code login..."
	aws sso login --profile "${AWS_PROFILE}" --use-device-code
fi

if ! aws sts get-caller-identity &>/dev/null; then
	echo "error: AWS credentials still not valid after login attempt." >&2
	exit 1
fi

mkdir -p "${DEST}"

# A single `aws s3 sync` already parallelizes internally (default 10 concurrent
# requests), but fanning out one sync per top-level route folder multiplies
# that across many processes for much higher aggregate throughput.
mapfile -t ROUTE_DIRS < <(
	aws s3 ls "s3://${BUCKET}/${PREFIX}/" | awk '{print $2}' | sed 's:/$::'
)

if [ ${#ROUTE_DIRS[@]} -eq 0 ]; then
	echo "No route folders found under s3://${BUCKET}/${PREFIX}/ - falling back to a single sync." >&2
	aws s3 sync "s3://${BUCKET}/${PREFIX}/" "${DEST}/"
else
	echo "Pulling ${#ROUTE_DIRS[@]} route folders from s3://${BUCKET}/${PREFIX}/ into ${DEST} (up to ${JOBS} in parallel)..."
	export BUCKET PREFIX DEST
	printf '%s\n' "${ROUTE_DIRS[@]}" | xargs -I{} -P "${JOBS}" bash -c '
		route="$1"
		aws s3 sync "s3://${BUCKET}/${PREFIX}/${route}/" "${DEST}/${route}/" --only-show-errors
		echo "done: ${route}"
	' _ {}
fi

echo "Done. Data pulled to ${DEST}"
