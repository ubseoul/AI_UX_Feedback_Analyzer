#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.headless true
