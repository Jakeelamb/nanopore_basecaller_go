#!/usr/bin/env bash
set -euo pipefail
export GIT_LFS_SKIP_SMUDGE=1
cd "$(dirname "$0")/.."
target="${1:-external/dorado-source}"
pin=8b8fc5d36a9c0baab262a743cb175b5878e38ca6
if [[ ! -d "$target/.git" ]]; then
  if [[ -d "$target" && -n "$(ls -A "$target")" ]]; then
    echo "Refusing nonempty destination: $target" >&2
    exit 1
  fi
  mkdir -p "$target"
  git -C "$target" init
  git -C "$target" remote add origin https://github.com/nanoporetech/dorado.git
  git -C "$target" fetch --depth=1 origin "$pin"
  git -C "$target" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "$target" rev-parse HEAD)" != "$pin" ]]; then
  echo "Existing source has the wrong commit; refusing to replace it" >&2
  exit 1
fi
if [[ -n "$(git -C "$target" status --porcelain --untracked-files=all)" ]]; then
  echo "Existing source has local changes; refusing to replace them" >&2
  exit 1
fi
git -C "$target" submodule update --init --depth=1 dorado/3rdparty/NVTX
printf 'Pinned source ready: %s\n' "$target"
