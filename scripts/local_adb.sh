#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TOOLS_ROOT="$REPO_ROOT/.local_tools"
DEB_ROOT="$TOOLS_ROOT/adb_debs"
EXTRACT_ROOT="$TOOLS_ROOT/adb_local"
ADB_ROOT="$EXTRACT_ROOT/usr"
ADB_BIN="$ADB_ROOT/bin/adb"
ADB_LIB="$ADB_ROOT/lib/x86_64-linux-gnu/android"

bootstrap_local_adb() {
  if [[ ! -d "$DEB_ROOT" ]]; then
    return 1
  fi

  shopt -s nullglob
  local debs=("$DEB_ROOT"/*.deb)
  shopt -u nullglob
  if [[ ${#debs[@]} -eq 0 ]]; then
    return 1
  fi

  rm -rf "$EXTRACT_ROOT"
  mkdir -p "$EXTRACT_ROOT"
  for deb in "${debs[@]}"; do
    dpkg-deb -x "$deb" "$EXTRACT_ROOT"
  done
}

if [[ ! -x "$ADB_BIN" ]]; then
  bootstrap_local_adb || {
    echo "local adb not found at $ADB_BIN" >&2
    echo "Cached adb packages not found at $DEB_ROOT." >&2
    exit 1
  }
fi

export LD_LIBRARY_PATH="$ADB_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ADB_BIN" "$@"
