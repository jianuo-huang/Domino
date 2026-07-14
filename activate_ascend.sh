#!/usr/bin/env bash

# Source this file so that Conda and CANN variables remain in the caller's shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run this script with: source ./activate_ascend.sh" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
if [[ -n "${DOMINO_HF_HOME:-}" ]]; then
  export HF_HOME="${DOMINO_HF_HOME}"
fi

if [[ "${DOMINO_ASCEND_ENV_ACTIVE:-0}" == "1" ]]; then
  return 0
fi

# Avoid inheriting packages from an unrelated workspace. Keep the variable
# defined but empty because CANN's setup script reads it under runners that use
# ``set -u``, then reconstructs it with only the toolkit paths it needs.
export PYTHONPATH=""

domino_conda_env="${DOMINO_CONDA_ENV:-domino-ascend}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${domino_conda_env}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available; install Conda before activating ${domino_conda_env}." >&2
    return 1
  fi

  domino_conda_base="$(conda info --base)" || return 1
  # shellcheck disable=SC1091
  source "${domino_conda_base}/etc/profile.d/conda.sh" || return 1
  conda activate "${domino_conda_env}" || return 1
fi

cann_env_script=""
cann_root=""
if [[ -n "${CANN_ENV_SCRIPT:-}" ]]; then
  if [[ ! -r "${CANN_ENV_SCRIPT}" ]]; then
    echo "Explicit CANN_ENV_SCRIPT is not readable: ${CANN_ENV_SCRIPT}" >&2
    return 1
  fi
  cann_env_script="${CANN_ENV_SCRIPT}"
elif [[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]]; then
  cann_root="${ASCEND_HOME_PATH}"
elif [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -d "${ASCEND_TOOLKIT_HOME}" ]]; then
  cann_root="${ASCEND_TOOLKIT_HOME}"
else
  machine_arch="$(uname -m)"
  cann_roots=()
  if [[ -n "${CANN_ROOT:-}" ]]; then
    cann_roots+=("${CANN_ROOT}")
  fi
  cann_roots+=(
    "/usr/local/Ascend/ascend-toolkit"
    "/usr/local/Ascend/ascend-toolkit/latest"
  )
  for candidate_root in "${cann_roots[@]}"; do
    for candidate_script in \
      "${candidate_root}/set_env.sh" \
      "${candidate_root}/${machine_arch}-linux/script/set_env.sh" \
      "${candidate_root}/${machine_arch}-linux/bin/setenv.bash" \
      "${candidate_root}/bin/setenv.bash"; do
      if [[ -r "${candidate_script}" ]]; then
        cann_root="${candidate_root}"
        cann_env_script="${candidate_script}"
        break 2
      fi
    done
  done
  if [[ -z "${cann_env_script}" ]]; then
    echo "Could not find a CANN environment script. Set CANN_ENV_SCRIPT or CANN_ROOT." >&2
    return 1
  fi
fi

if [[ -n "${cann_env_script}" ]]; then
  # shellcheck disable=SC1090
  source "${cann_env_script}" || return 1
fi

export DOMINO_ASCEND_ENV_ACTIVE=1
active_cann_root="${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-${cann_root}}}"
echo "Activated ${domino_conda_env} with CANN at ${active_cann_root}."
if [[ -n "${HF_HOME:-}" ]]; then
  echo "Hugging Face cache: ${HF_HOME}"
else
  echo "Hugging Face cache: Hugging Face default"
fi
