#!/usr/bin/env bash

# Source this file so that Conda and CANN variables remain in the caller's shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run this script with: source ./activate_ascend.sh" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export HF_HOME="${DOMINO_HF_HOME:-/mnt/nvme0/zhujiayi/.cache/huggingface}"

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

cann_root="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/8.0.1}"
cann_env_script="${CANN_ENV_SCRIPT:-${cann_root}/aarch64-linux/script/set_env.sh}"
if [[ ! -r "${cann_env_script}" ]]; then
  echo "CANN 8.0.1 setup script not found: ${cann_env_script}" >&2
  return 1
fi

# shellcheck disable=SC1090
source "${cann_env_script}" || return 1

echo "Activated ${domino_conda_env} with CANN at ${ASCEND_TOOLKIT_HOME}."
echo "Hugging Face cache: ${HF_HOME}"
