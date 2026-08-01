# Shared AWS CLI check, SSO login prompt, and ECR docker login for SIL images.
# Sourced by container_sil_run*.sh. Caller must set SCRIPT_DIR to the docker/ directory.

SIL_ECR_DOMAIN="${SIL_ECR_DOMAIN:-458214780330.dkr.ecr.us-west-2.amazonaws.com}"
SIL_ECR_REGION="${SIL_ECR_REGION:-us-west-2}"

if [ -z "${SCRIPT_DIR:-}" ]; then
	echo "error: SCRIPT_DIR must be set before sourcing sil_ecr_auth.sh" >&2
	return 1 2>/dev/null || exit 1
fi

if [ -f "${SCRIPT_DIR}/aws_config.ini" ]; then
	export AWS_CONFIG_FILE="${SCRIPT_DIR}/aws_config.ini"
	if [ -z "${AWS_PROFILE:-}" ]; then
		_sil_plist=""
		if command -v aws &>/dev/null; then
			_sil_plist=$(aws configure list-profiles 2>/dev/null || true)
		fi
		if echo "${_sil_plist}" | grep -qx 'sil-ecr-readonly'; then
			export AWS_PROFILE=sil-ecr-readonly
		elif echo "${_sil_plist}" | grep -qx 'default'; then
			export AWS_PROFILE=default
		elif [ -n "$(echo "${_sil_plist}" | head -n1 | tr -d '\r')" ]; then
			export AWS_PROFILE="$(echo "${_sil_plist}" | head -n1)"
			echo "Using AWS_PROFILE=${AWS_PROFILE} from docker/aws_config.ini" >&2
		fi
		unset _sil_plist
	fi
fi

for _sil_utils in \
	"${SCRIPT_DIR}/../utils/cb_bash_functions.sh" \
	"${SCRIPT_DIR}/../../utils/cb_bash_functions.sh" \
	"${SCRIPT_DIR}/../apollo/utils/cb_bash_functions.sh"; do
	if [ -f "${_sil_utils}" ]; then
		# shellcheck source=/dev/null
		source "${_sil_utils}"
		break
	fi
done
unset _sil_utils

if ! declare -F cb_ask >/dev/null 2>&1; then
	cb_ask() {
		read -r -p "$1 [y/N] " _sil_r || true
		[[ "${_sil_r}" =~ ^[yY] ]]
	}
fi

: "${AWS_REJECT_MESSAGE:=Aborted: AWS setup is required to pull the SIL image.}"

sil_ensure_aws_and_ecr() {
	set -o pipefail

	if ! command -v aws &>/dev/null; then
		echo "AWS CLI could not be found." >&2
		if cb_ask "Would you like to install the AWS CLI v2"; then
			_tmp="${TMPDIR:-/tmp}"
			pushd "${_tmp}" >/dev/null || exit 1
			SYSTEM_ARCH=$(uname -m)
			curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${SYSTEM_ARCH}.zip" -o "awscliv2.zip"
			unzip -q -o awscliv2.zip
			sudo ./aws/install
			popd >/dev/null || true
		else
			echo "${AWS_REJECT_MESSAGE}" >&2
			set +o pipefail
			return 1
		fi
	fi

	if ! command -v aws &>/dev/null; then
		echo "error: AWS CLI still not available after install attempt." >&2
		set +o pipefail
		return 1
	fi

	if ! docker info &>/dev/null; then
		echo "error: cannot talk to the Docker daemon (running? user in docker group?)." >&2
		set +o pipefail
		return 1
	fi

	if ! aws sts get-caller-identity &>/dev/null; then
		echo "Not logged in to AWS. If you need access, ask your AWS admin." >&2
		if cb_ask "Would you like to run aws sso login now (required to proceed)"; then
			aws sso login --use-device-code
		else
			echo "${AWS_REJECT_MESSAGE}" >&2
			set +o pipefail
			return 1
		fi
	fi

	if ! aws sts get-caller-identity &>/dev/null; then
		echo "error: AWS credentials still not valid after login attempt." >&2
		set +o pipefail
		return 1
	fi

	echo "Logging Docker in to ${SIL_ECR_DOMAIN} ..."
	if ! aws ecr get-login-password --region "${SIL_ECR_REGION}" | docker login --username AWS --password-stdin "${SIL_ECR_DOMAIN}"; then
		echo "error: docker login failed. Check IAM/SSO for ECR and registry account in ${SIL_ECR_DOMAIN}." >&2
		set +o pipefail
		return 1
	fi

	set +o pipefail
	return 0
}

# Ensures apollo-nav-standalone:latest exists locally for docker run (full ECR ref).
sil_ensure_standalone_image_local() {
	local full="${SIL_ECR_DOMAIN}/apollo-nav-standalone:latest"

	if docker image inspect "${full}" &>/dev/null; then
		return 0
	fi
	if docker image inspect "apollo-nav-standalone:latest" &>/dev/null; then
		docker tag "apollo-nav-standalone:latest" "${full}"
		return 0
	fi

	echo "SIL base image not found locally; pulling apollo-nav-standalone:latest ..." >&2
	if [ -f "${SCRIPT_DIR}/pull_apollo_nav_standalone.sh" ] && [ -f "${SCRIPT_DIR}/aws_config.ini" ]; then
		bash "${SCRIPT_DIR}/pull_apollo_nav_standalone.sh" --release-tag=latest
	else
		if [ ! -f "${SCRIPT_DIR}/aws_config.ini" ]; then
			echo "Note: docker/aws_config.ini not found; using docker pull with your current AWS session." >&2
		fi
		docker pull "${full}"
	fi
}

# If a container with this name exists, attach (running) or start -ai (stopped); exec replaces shell.
# Returns 1 if no such container (caller should docker run).
sil_try_reuse_named_container() {
	local name="$1"
	if ! docker inspect "$name" &>/dev/null; then
		return 1
	fi
	local running
	running=$(docker inspect -f '{{.State.Running}}' "$name")
	if [ "$running" = "true" ]; then
		echo "Container ${name} is already running; attaching (Ctrl+P then Ctrl+Q to detach without stopping)." >&2
		exec docker attach "$name"
	else
		echo "Starting existing container ${name} ..." >&2
		exec docker start -ai "$name"
	fi
}
