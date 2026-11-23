#!/usr/bin/env bash
# Install the app as a uv tool and add a launcher to the application menu and the desktop.
# Usage: scripts/install_desktop.sh [--editable]
#   --editable  run straight from this checkout, so code changes apply without reinstalling
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ID="inspire-hand-teleop"

if [[ "${1:-}" == "--editable" ]]; then
    uv tool install --force --editable "$REPO"
else
    uv tool install --force "$REPO"
fi

BIN="$(uv tool dir --bin)/inspire-teleop"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
mkdir -p "$ICON_DIR" "$APPS_DIR" "$DESKTOP_DIR"
cp "$REPO/inspire_teleop/assets/icon.svg" "$ICON_DIR/$APP_ID.svg"

cat > "$APPS_DIR/$APP_ID.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Inspire Hand Teleop
Comment=Control the Inspire RH56 robot hand with your webcam
Exec="$BIN" app
Icon=$APP_ID
Terminal=false
Categories=Science;Engineering;
StartupWMClass=inspire-hand-teleop
EOF
chmod +x "$APPS_DIR/$APP_ID.desktop"

cp "$APPS_DIR/$APP_ID.desktop" "$DESKTOP_DIR/$APP_ID.desktop"
chmod +x "$DESKTOP_DIR/$APP_ID.desktop"
gio set "$DESKTOP_DIR/$APP_ID.desktop" metadata::trusted true 2>/dev/null || true
update-desktop-database "$APPS_DIR" 2>/dev/null || true

echo "Installed: $BIN"
echo "Launcher:  $APPS_DIR/$APP_ID.desktop"
echo "Desktop:   $DESKTOP_DIR/$APP_ID.desktop"
