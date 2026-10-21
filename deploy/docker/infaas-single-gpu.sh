#!/usr/bin/env bash
# The selected Docker daemon/context is never changed by this launcher.
set -euo pipefail
INFAAS_SINGLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$INFAAS_SINGLE_ROOT"
exec python3 -m infaas.single_gpu.deploy "$@"
