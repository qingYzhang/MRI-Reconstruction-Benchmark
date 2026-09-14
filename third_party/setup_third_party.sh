#!/usr/bin/env bash

set -euo pipefail

ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

EXTERNAL="${ROOT}/external"

LAPS_DIR="${EXTERNAL}/laps_original"
SHAMAEI_DIR="${EXTERNAL}/shamaei_original"

LAPS_COMMIT="ca1b5cc8d0d24b164a848c6fbd06b3fc5ec7d99b"
SHAMAEI_COMMIT="8ef48bc8a9956744c28bce068fb23aff0c268f2a"

LAPS_PATCH="${ROOT}/third_party/patches/laps_diffusers_compat.patch"

mkdir -p "${EXTERNAL}"

echo "============================================================"
echo "Setting up third-party repositories"
echo "ROOT=${ROOT}"
echo "============================================================"


setup_repo() {

    local name="$1"
    local url="$2"
    local dir="$3"
    local commit="$4"

    echo
    echo "------------------------------------------------------------"
    echo "${name}"
    echo "------------------------------------------------------------"

    if [ ! -e "${dir}" ]; then

        echo "Cloning ${url}"
        git clone "${url}" "${dir}"

        git -C "${dir}" checkout "${commit}"

        echo "Created:"
        echo "  ${dir}"

    else

        if ! git -C "${dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            echo "ERROR: ${dir} exists but is not a Git repository."
            exit 1
        fi

        current="$(
            git -C "${dir}" rev-parse HEAD
        )"

        if [ "${current}" != "${commit}" ]; then
            echo "ERROR: ${name} is at the wrong commit."
            echo "Expected: ${commit}"
            echo "Current:  ${current}"
            echo
            echo "Refusing to modify an existing checkout automatically."
            exit 1
        fi

        echo "Existing checkout found."
        echo "Commit matches:"
        echo "  ${commit}"

    fi
}


setup_repo \
    "LAPS" \
    "https://github.com/SetsompopLab/LAPS.git" \
    "${LAPS_DIR}" \
    "${LAPS_COMMIT}"


#
# Initialize LAPS submodules only when needed.
#
if [ ! -e "${LAPS_DIR}/submodules/diffusers/.git" ] \
   && [ ! -f "${LAPS_DIR}/submodules/diffusers/.git" ]; then

    echo
    echo "Initializing LAPS submodules..."

    git -C "${LAPS_DIR}" \
        submodule update \
        --init \
        --recursive
fi


#
# Apply the compatibility patch inside the vendored
# diffusers submodule.
#
DIFFUSERS_DIR="${LAPS_DIR}/submodules/diffusers"

if git -C "${DIFFUSERS_DIR}" \
    apply --check -p3 "${LAPS_PATCH}" \
    >/dev/null 2>&1; then

    echo
    echo "Applying LAPS diffusers compatibility patch..."

    git -C "${DIFFUSERS_DIR}" \
        apply -p3 "${LAPS_PATCH}"

elif git -C "${DIFFUSERS_DIR}" \
    apply --reverse --check -p3 "${LAPS_PATCH}" \
    >/dev/null 2>&1; then

    echo
    echo "LAPS compatibility patch is already applied."

else

    echo
    echo "ERROR:"
    echo "LAPS compatibility patch cannot be applied cleanly"
    echo "and does not appear to already be applied."
    exit 1

fi


setup_repo \
    "Shamaei et al." \
    "https://github.com/amirshamaei/longitudinal-mri-deep-recon.git" \
    "${SHAMAEI_DIR}" \
    "${SHAMAEI_COMMIT}"


echo
echo "============================================================"
echo "Third-party setup complete"
echo "============================================================"

echo
echo "LAPS:"
git -C "${LAPS_DIR}" rev-parse HEAD

echo
echo "Shamaei:"
git -C "${SHAMAEI_DIR}" rev-parse HEAD

echo
echo "For LAPS experiments use:"
echo "  export PYTHONPATH=\"code:\$PWD/external/laps_original/src\""

echo
echo "For Shamaei experiments use:"
echo "  export PYTHONPATH=\"code\""
