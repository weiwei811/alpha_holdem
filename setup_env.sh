#!/usr/bin/env sh
set -eu
cd -- "$(dirname -- "$0")"
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'from agi.devices import resolve_device; print("Available device:", resolve_device("auto"))'
