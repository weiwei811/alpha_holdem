#!/bin/bash
pip install -q gymnasium
python3 train_league.py --conf confs/nl_holdem.py --sp 0.0 --upwin 1.0 --gap=500 --league_tracker_n 1000
