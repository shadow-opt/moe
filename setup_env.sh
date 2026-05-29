#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '\n\033[1;31m[error]\033[0m Please run this script with: bash setup_env.sh\n' >&2
    printf 'Do not use "source setup_env.sh"; this installer uses conda run and does not need to activate the environment.\n' >&2
    return 1 2>/dev/null || exit 1
fi

ENV_NAME="${ENV_NAME:-moe}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.3.1}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$HOME/miniconda3}"
NETWORK_TURBO="${NETWORK_TURBO:-/etc/network_turbo}"
REPO_URL="${REPO_URL:-https://github.com/shadow-opt/moe.git}" # TODO: update this URL to the actual repository if different
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ISAAC_GYM_URL="${ISAAC_GYM_URL:-https://developer.nvidia.com/isaac-gym-preview-4}"
ISAAC_GYM_ARCHIVE="${ISAAC_GYM_ARCHIVE:-}"
DOWNLOAD_ISAAC_GYM=1
INSTALL_MINICONDA=1
CONDA_CHECK_TIMEOUT="${CONDA_CHECK_TIMEOUT:-60}"

usage() {
    cat <<EOF
Usage: bash setup_env.sh [options]

Options:
  --env NAME                 Conda environment name. Default: ${ENV_NAME}
  --python VERSION           Python version. Default: ${PYTHON_VERSION}
  --cuda VERSION             PyTorch CUDA version. Default: ${CUDA_VERSION}
  --repo-dir PATH            Repository path. Default: script directory.
  --repo-url URL             Git URL used when --repo-dir does not contain this repo.
  --isaacgym PATH            Isaac Gym Preview 4 archive or extracted directory.
  --no-download-isaacgym     Do not try to download Isaac Gym when missing.
  --no-install-conda         Fail instead of installing Miniconda when conda is missing.
  -h, --help                 Show this help.

Environment variables with the same names can also be used:
  ENV_NAME, PYTHON_VERSION, CUDA_VERSION, ISAAC_GYM_ARCHIVE, ISAAC_GYM_URL,
  CONDA_PREFIX_DIR, NETWORK_TURBO, REPO_DIR, REPO_URL
EOF
}

log() {
    printf '\n\033[1;32m[setup]\033[0m %s\n' "$*" >&2
}

warn() {
    printf '\n\033[1;33m[warn]\033[0m %s\n' "$*" >&2
}

die() {
    printf '\n\033[1;31m[error]\033[0m %s\n' "$*" >&2
    exit 1
}

run() {
    printf '\033[1;30m+\033[0m' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    "$@"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            ENV_NAME="${2:?missing value for --env}"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="${2:?missing value for --python}"
            shift 2
            ;;
        --cuda)
            CUDA_VERSION="${2:?missing value for --cuda}"
            shift 2
            ;;
        --repo-dir)
            REPO_DIR="${2:?missing value for --repo-dir}"
            shift 2
            ;;
        --repo-url)
            REPO_URL="${2:?missing value for --repo-url}"
            shift 2
            ;;
        --isaacgym)
            ISAAC_GYM_ARCHIVE="${2:?missing value for --isaacgym}"
            shift 2
            ;;
        --no-download-isaacgym)
            DOWNLOAD_ISAAC_GYM=0
            shift
            ;;
        --no-install-conda)
            INSTALL_MINICONDA=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

ensure_repo() {
    if [[ -f "${REPO_DIR}/setup.py" && -d "${REPO_DIR}/rsl_rl" && -d "${REPO_DIR}/legged_gym" ]]; then
        REPO_DIR="$(cd "${REPO_DIR}" && pwd)"
        return
    fi

    if [[ -e "${REPO_DIR}" ]]; then
        die "${REPO_DIR} exists, but it does not look like this repository. Pass --repo-dir /path/to/moe."
    fi

    need_cmd git
    log "Cloning repository to ${REPO_DIR}"
    run git clone "${REPO_URL}" "${REPO_DIR}"
    REPO_DIR="$(cd "${REPO_DIR}" && pwd)"
}

find_conda() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
    elif [[ -x "${CONDA_PREFIX_DIR}/bin/conda" ]]; then
        printf '%s\n' "${CONDA_PREFIX_DIR}/bin/conda"
    else
        return 1
    fi
}

install_miniconda() {
    need_cmd wget
    log "Installing Miniconda to ${CONDA_PREFIX_DIR}"
    mkdir -p "${CONDA_PREFIX_DIR}"
    local installer="${CONDA_PREFIX_DIR}/miniconda.sh"
    run wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "${installer}"
    run bash "${installer}" -b -u -p "${CONDA_PREFIX_DIR}"
    rm -f "${installer}"
}

ensure_conda() {
    local conda_exe
    if conda_exe="$(find_conda)"; then
        printf '%s\n' "${conda_exe}"
        return
    fi

    [[ "${INSTALL_MINICONDA}" == "1" ]] || die "conda not found. Install Conda first or rerun without --no-install-conda."
    install_miniconda >&2
    find_conda || die "Miniconda installation finished, but conda is still not available."
}

conda_env_exists() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "${CONDA_CHECK_TIMEOUT}" "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
    else
        "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
    fi
}

conda_run() {
    "${CONDA_EXE}" run -n "${ENV_NAME}" "$@"
}

ensure_conda_env() {
    if conda_env_exists; then
        log "Conda environment '${ENV_NAME}' already exists"
    else
        log "Conda environment '${ENV_NAME}' not found; creating it now"
        log "Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
        run "${CONDA_EXE}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
    fi
}

ensure_pytorch() {
    log "Installing PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION}"
    run "${CONDA_EXE}" install -y -n "${ENV_NAME}" \
        "pytorch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        "pytorch-cuda=${CUDA_VERSION}" \
        -c pytorch -c nvidia
}

isaac_python_dir_from_path() {
    local path="$1"

    if [[ -d "${path}/python" && -f "${path}/python/setup.py" ]]; then
        printf '%s\n' "${path}/python"
    elif [[ -d "${path}/isaacgym/python" && -f "${path}/isaacgym/python/setup.py" ]]; then
        printf '%s\n' "${path}/isaacgym/python"
    elif [[ -f "${path}/setup.py" && "$(basename "${path}")" == "python" ]]; then
        printf '%s\n' "${path}"
    else
        return 1
    fi
}

extract_isaacgym_archive() {
    local archive="$1"
    local target="${REPO_DIR}/.isaacgym_extract"

    [[ -f "${archive}" ]] || die "Isaac Gym archive not found: ${archive}"
    mkdir -p "${target}"

    case "${archive}" in
        *.tar.gz|*.tgz|*.tar)
            tar -tf "${archive}" >/dev/null 2>&1 || die "Invalid tar archive: ${archive}"
            log "Extracting Isaac Gym archive"
            tar -xf "${archive}" -C "${target}"
            ;;
        *.zip)
            need_cmd unzip
            log "Extracting Isaac Gym archive"
            unzip -q -o "${archive}" -d "${target}"
            ;;
        *)
            die "Unsupported Isaac Gym archive type: ${archive}"
            ;;
    esac

    local candidate
    candidate="$(find "${target}" -path '*/isaacgym/python/setup.py' -o -path '*/python/setup.py' | head -n 1 || true)"
    [[ -n "${candidate}" ]] || die "Could not find isaacgym/python/setup.py after extracting ${archive}"
    dirname "${candidate}"
}

download_isaacgym() {
    need_cmd wget
    local archive="${REPO_DIR}/IsaacGym_Preview_4_Package.tar.gz"

    log "Downloading Isaac Gym Preview 4"
    if ! wget -O "${archive}" -L "${ISAAC_GYM_URL}"; then
        rm -f "${archive}"
        die "Isaac Gym download failed. Download it manually from https://developer.nvidia.com/isaac-gym and rerun with --isaacgym /path/to/archive."
    fi

    ISAAC_GYM_ARCHIVE="${archive}"
}

find_isaacgym_python_dir() {
    local candidate

    if [[ -n "${ISAAC_GYM_ARCHIVE}" ]]; then
        if [[ -d "${ISAAC_GYM_ARCHIVE}" ]]; then
            isaac_python_dir_from_path "${ISAAC_GYM_ARCHIVE}" || die "Could not find isaacgym/python/setup.py in ${ISAAC_GYM_ARCHIVE}"
            return
        fi

        extract_isaacgym_archive "${ISAAC_GYM_ARCHIVE}"
        return
    fi

    if candidate="$(isaac_python_dir_from_path "${REPO_DIR}/isaacgym" 2>/dev/null)"; then
        printf '%s\n' "${candidate}"
        return
    fi

    candidate="$(find "${REPO_DIR}" -maxdepth 2 -type f \
        \( -name 'IsaacGym*.tar.gz' -o -name 'isaac-gym*.tar.gz' -o -name 'isaacgym*.tar.gz' -o -name 'IsaacGym*.zip' \) \
        | head -n 1 || true)"
    if [[ -n "${candidate}" ]]; then
        extract_isaacgym_archive "${candidate}"
        return
    fi

    if [[ "${DOWNLOAD_ISAAC_GYM}" == "1" ]]; then
        download_isaacgym
        extract_isaacgym_archive "${ISAAC_GYM_ARCHIVE}"
        return
    fi

    die "Isaac Gym not found. Put the extracted 'isaacgym' directory in the repo, or rerun with --isaacgym /path/to/IsaacGym_Preview_4_Package.tar.gz."
}

install_editable_packages() {
    local isaac_python_dir="$1"

    log "Installing Isaac Gym"
    run conda_run python -m pip install -e "${isaac_python_dir}"

    log "Installing local rsl_rl"
    run conda_run python -m pip install -e "${REPO_DIR}/rsl_rl"

    log "Installing this repository"
    run conda_run python -m pip install -e "${REPO_DIR}"
}

verify_install() {
    log "Verifying installed packages"
    conda_run python -c '
import importlib
import importlib.util

for name in ("isaacgym", "rsl_rl", "legged_gym", "mujoco"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ModuleNotFoundError(name)
    print(f"{name}: {spec.origin or spec.submodule_search_locations}")

torch = importlib.import_module("torch")
print(f"torch: {torch.__version__}")
'
}

print_activation_hint() {
    local conda_base
    conda_base="$("${CONDA_EXE}" info --base)"

    if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
        log "Done. Activate the environment with: source ${conda_base}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
    else
        log "Done. Activate the environment with: conda activate ${ENV_NAME}"
    fi
}

main() {
    if [[ -f "${NETWORK_TURBO}" ]]; then
        log "Sourcing ${NETWORK_TURBO}"
        # shellcheck disable=SC1090
        source "${NETWORK_TURBO}" || warn "Failed to source ${NETWORK_TURBO}; continuing without it"
    fi

    ensure_repo

    CONDA_EXE="$(ensure_conda)"
    export CONDA_EXE
    log "Using Conda: ${CONDA_EXE}"

    ensure_conda_env
    ensure_pytorch

    local isaac_python_dir
    isaac_python_dir="$(find_isaacgym_python_dir)"
    install_editable_packages "${isaac_python_dir}"
    verify_install

    print_activation_hint
}

main "$@"
