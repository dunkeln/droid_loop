#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/droid_loop

sudo apt-get update
sudo apt-get install -y git nginx ffmpeg python3-pip curl

if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

if [ ! -d "$APP_DIR" ]; then
  sudo mkdir -p "$APP_DIR"
  sudo chown -R "$USER":"$USER" "$APP_DIR"
  git clone https://github.com/dunkeln/droid_loop.git "$APP_DIR"
fi

cd "$APP_DIR"
if [ ! -f .env ]; then
  cp .env.example .env
fi
uv sync
cd ui
npm install
npm run build:app
cd ..

sudo cp deploy/lightsail/droid-loop.service /etc/systemd/system/droid-loop.service
sudo cp deploy/lightsail/nginx.conf /etc/nginx/sites-available/droid-loop
sudo ln -sf /etc/nginx/sites-available/droid-loop /etc/nginx/sites-enabled/droid-loop
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable droid-loop
sudo systemctl restart droid-loop
sudo systemctl restart nginx
