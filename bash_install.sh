#!/usr/bin/env bash
module load python3/3.9.6
python3 -m virtualenv -p3.9 venv
source venv/bin/activate
python3 -m pip install mosaic-library