#!/usr/bin/env bash

set -Eeuo pipefail

SERVICE_NAME="hak-gemini-binance-trader"
EXPECTED_APP_DIR="/root/hak-gemini-binance-trader"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR"
VENV_DIR="$APP_DIR/venv"
REQUIREMENTS_FILE="$APP_DIR/requirements.txt"
ENV_EXAMPLE_FILE="$APP_DIR/.env.example"
ENV_FILE="$APP_DIR/.env"
SERVICE_SOURCE="$APP_DIR/${SERVICE_NAME}.service"
SERVICE_TARGET="/etc/systemd/system/${SERVICE_NAME}.service"

NO_START=0
SKIP_APT=0

trap 'printf "[ERROR] setup_linode_systemd.sh failed near line %s\n" "${BASH_LINENO[0]}" >&2' ERR

log() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  sudo bash setup_linode_systemd.sh [--no-start] [--skip-apt]

Options:
  --no-start   Install and enable the service, but do not start it.
  --skip-apt   Skip apt-based package installation.
  -h, --help   Show this help message.
EOF
}

while (($# > 0)); do
  case "$1" in
    --no-start)
      NO_START=1
      ;;
    --skip-apt)
      SKIP_APT=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
  shift
done

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    fail "Run this script as root or with sudo."
  fi
}

require_systemd() {
  if ! command -v systemctl >/dev/null 2>&1; then
    fail "systemctl is not available on this VPS."
  fi
}

require_expected_layout() {
  if [[ "$APP_DIR" != "$EXPECTED_APP_DIR" ]]; then
    fail "This script expects the repository at $EXPECTED_APP_DIR, but found $APP_DIR."
  fi

  [[ -f "$REQUIREMENTS_FILE" ]] || fail "Missing $REQUIREMENTS_FILE"
  [[ -f "$SERVICE_SOURCE" ]] || fail "Missing $SERVICE_SOURCE"
  [[ -f "$APP_DIR/main.py" ]] || fail "Missing $APP_DIR/main.py"
  [[ -f "$APP_DIR/setting.yaml" ]] || fail "Missing $APP_DIR/setting.yaml"
}

apt_install() {
  if [[ "$SKIP_APT" -eq 1 ]]; then
    warn "Skipping apt installation because --skip-apt was provided."
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    fail "apt-get is not available. Install Python manually, then rerun with --skip-apt."
  fi

  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y "$@"
}

ensure_base_packages() {
  if command -v python3 >/dev/null 2>&1; then
    log "python3 is already installed."
    return
  fi

  log "Installing base Python packages with apt..."
  apt_install python3 python3-venv ca-certificates
}

select_python_bin() {
  local candidate
  for candidate in python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

ensure_python() {
  ensure_base_packages

  PYTHON_BIN="$(select_python_bin)" || fail "Unable to find python3 after installation."
  PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  log "Using Python interpreter: $PYTHON_BIN ($PYTHON_VERSION)"

  if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)'; then
    warn "Python 3.13+ is recommended by the README. Continuing with $PYTHON_VERSION."
  fi
}

ensure_matching_venv_package() {
  local python_mm

  python_mm="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  log "Installing matching venv package for Python $python_mm..."

  if [[ "$SKIP_APT" -eq 1 ]]; then
    fail "Virtual environment creation failed and --skip-apt is enabled."
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    fail "Virtual environment creation failed and apt-get is not available."
  fi

  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  if ! apt-get install -y "python${python_mm}-venv"; then
    apt-get install -y python3-venv
  fi
}

ensure_virtualenv() {
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    log "Reusing existing virtual environment at $VENV_DIR"
    return
  fi

  log "Creating virtual environment at $VENV_DIR"
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    ensure_matching_venv_package
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
}

install_python_dependencies() {
  log "Installing Python dependencies into the virtual environment..."
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"
}

prepare_runtime_dirs() {
  log "Ensuring runtime directories exist..."
  mkdir -p "$APP_DIR/log" "$APP_DIR/db"
}

prepare_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    log ".env already exists. Leaving it untouched."
  elif [[ -f "$ENV_EXAMPLE_FILE" ]]; then
    log "Creating .env from .env.example"
    cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
  else
    warn "No .env.example found. Skipping .env bootstrap."
    return
  fi

  chmod 600 "$ENV_FILE"
}

get_env_value() {
  local key="$1"
  local line
  local value

  if [[ ! -f "$ENV_FILE" ]]; then
    return 1
  fi

  line="$(grep -E "^${key}=" "$ENV_FILE" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    return 1
  fi

  value="${line#*=}"
  value="${value#\"}"
  value="${value%\"}"
  value="${value#\'}"
  value="${value%\'}"
  printf '%s' "$value"
}

is_placeholder_value() {
  case "$1" in
    ""|YOUR_*|your_*|CHANGEME|changeme)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

required_env_ready() {
  local key
  local value
  local missing=()

  for key in BINANCE_API_KEY BINANCE_API_SECRET GEMINI_API_KEY; do
    value="$(get_env_value "$key" || true)"
    if is_placeholder_value "$value"; then
      missing+=("$key")
    fi
  done

  if ((${#missing[@]} > 0)); then
    warn "Fill in real values in $ENV_FILE before starting the bot: ${missing[*]}"
    return 1
  fi

  return 0
}

install_service_unit() {
  log "Installing systemd service unit to $SERVICE_TARGET"
  install -m 0644 "$SERVICE_SOURCE" "$SERVICE_TARGET"
  systemctl daemon-reload
  systemctl enable "$SERVICE_NAME"
}

start_service_if_ready() {
  if [[ "$NO_START" -eq 1 ]]; then
    log "Skipping service start because --no-start was provided."
    return
  fi

  if ! required_env_ready; then
    warn "Service was installed and enabled, but not started because required API keys are not ready."
    return
  fi

  log "Starting or restarting $SERVICE_NAME.service"
  if ! systemctl restart "$SERVICE_NAME"; then
    systemctl --no-pager --full status "$SERVICE_NAME" || true
    fail "Service failed to start. Review the status output above."
  fi

  systemctl --no-pager --full status "$SERVICE_NAME" || true
}

print_summary() {
  cat <<EOF

[INFO] Setup complete.
[INFO] Service file: $SERVICE_TARGET
[INFO] App directory: $APP_DIR
[INFO] Virtualenv: $VENV_DIR

[INFO] Helpful commands:
  systemctl status ${SERVICE_NAME} --no-pager
  systemctl restart ${SERVICE_NAME}
  journalctl -u ${SERVICE_NAME} -f
  tail -f ${APP_DIR}/log/ai_trader.log
EOF
}

main() {
  require_root
  require_systemd
  require_expected_layout
  ensure_python
  ensure_virtualenv
  install_python_dependencies
  prepare_runtime_dirs
  prepare_env_file
  install_service_unit
  start_service_if_ready
  print_summary
}

main "$@"
