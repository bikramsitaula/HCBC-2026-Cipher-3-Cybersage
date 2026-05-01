#!/bin/bash
set -e

export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
cd "$(dirname "$0")/backend"

echo "Installing dependencies..."
# Bootstrap pip if missing
if ! python3 -m pip --version &>/dev/null; then
  curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --user --break-system-packages
fi

python3 -m pip install -r requirements.txt --break-system-packages -q

echo "Starting CyberSage backend on port 8000..."
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
