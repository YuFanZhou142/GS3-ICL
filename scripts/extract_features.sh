#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/extract_meld_features.py --audio --visual --split all "$@"
