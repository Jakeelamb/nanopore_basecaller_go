#!/usr/bin/env bash
set -euo pipefail
exec go run ./verification/subread/check "$@"
