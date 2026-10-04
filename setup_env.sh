#!/usr/bin/env sh
set -eu
python3.11 -m venv .venv
requirements=requirements.txt
if [ "$(uname -s)" = Darwin ]; then
    case "$(uname -m)" in
        arm64) requirements=requirements-mac-arm64.txt ;;
        x86_64) requirements=requirements-mac-intel.txt ;;
    esac
fi
.venv/bin/python -m pip install -r "$requirements"
.venv/bin/python -c 'from agi.devices import resolve_device; print("Available device:", resolve_device("auto"))'
