#!/usr/bin/env sh
set -eu
exec .venv/bin/python train_league.py --device auto "$@"
