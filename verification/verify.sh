#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u LD_PRELOAD -u LD_AUDIT \
  GOWORK=off go run ./cmd/portcheck verify "$@"
