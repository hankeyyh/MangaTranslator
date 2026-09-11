#!/bin/bash
# MangaTranslator Modal phase-1 deploy helper.
#
# Usage:
#   ./deploy/deploy.sh setup
#   ./deploy/deploy.sh deploy
#   ./deploy/deploy.sh models
#   ./deploy/deploy.sh test [fast|precise|health] [-i image ...]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
}

resolve_modal() {
    if command -v modal >/dev/null 2>&1; then
        MODAL_BIN="$(command -v modal)"
        return
    fi
    local candidate="$HOME/.pyenv/versions/manga-image-translator/bin/modal"
    if [ -x "$candidate" ]; then
        MODAL_BIN="$candidate"
        return
    fi
    print_error "Modal CLI not found. Install with: pip install modal && modal token new"
    exit 1
}

PROD_ENV_FILE=".env.prod"
APP_NAME="manga-translator-mt"
SECRET_NAME="manga-translator-mt-env"

setup() {
    print_header "Modal Setup"
    resolve_modal

    if [ ! -f "$PROD_ENV_FILE" ]; then
        KEY="$(openssl rand -hex 32)"
        cat > "$PROD_ENV_FILE" <<EOF
MT_API_KEY=${KEY}
DEEPL_AUTH_KEY=
DEEPSEEK_API_KEY=
DEEPSEEK_API_BASE=https://api.deepseek.com
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
HF_TOKEN=
EOF
        print_success "Wrote ${PROD_ENV_FILE} with a new MT_API_KEY"
        print_warning "Fill DEEPL_AUTH_KEY, DEEPSEEK_API_KEY, SUPABASE_* and HF_TOKEN in ${PROD_ENV_FILE}"
    fi

    if ! grep -q "^MT_API_KEY=" "$PROD_ENV_FILE"; then
        KEY="$(openssl rand -hex 32)"
        echo "MT_API_KEY=$KEY" >> "$PROD_ENV_FILE"
        print_success "MT_API_KEY added to ${PROD_ENV_FILE}"
    fi

    print_info "Creating Modal secret ${SECRET_NAME} from ${PROD_ENV_FILE}"
    "$MODAL_BIN" secret create "$SECRET_NAME" --from-dotenv "$PROD_ENV_FILE" --force
    print_success "Setup completed"
    print_info "Next: ./deploy/deploy.sh deploy"
}

deploy() {
    print_header "Deploying ${APP_NAME}"
    resolve_modal
    "$MODAL_BIN" deploy deploy/modal_app.py
    print_success "Deploy finished"
    print_info "Next: ./deploy/deploy.sh models"
}

download_models() {
    print_header "Downloading models to Volume"
    resolve_modal
    "$MODAL_BIN" run deploy/modal_app.py::download_models
    print_success "Models ready"
}

run_tests() {
    local preset="fast"
    local image_args=()
    if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
        preset="$1"
        shift
    fi
    while [ $# -gt 0 ]; do
        case "$1" in
            -i|--image)
                image_args+=("--image" "$2")
                shift 2
                ;;
            *)
                print_warning "Ignoring extra arg: $1"
                shift
                ;;
        esac
    done

    print_header "Smoke test (${preset})"
    resolve_modal
    local username
    username="$("$MODAL_BIN" profile current)"
    local url="https://${username}--${APP_NAME}-web.modal.run"
    print_info "URL: $url"

    if [ -f "$PROD_ENV_FILE" ]; then
        # shellcheck disable=SC1090
        set -a
        # Modal dotenv files may contain spaces around '='.
        eval "$(python3 - "$PROD_ENV_FILE" <<'PY'
from pathlib import Path
import shlex
import sys
path = Path(sys.argv[1])
for raw in path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().strip("'").strip('"')
    if key == "MT_API_KEY":
        print(f"export MT_API_KEY={shlex.quote(value)}")
PY
)"
        set +a
    fi

    python3 deploy/smoke_test.py --url "$url" --preset "$preset" ${image_args[@]+"${image_args[@]}"}
}

view_logs() {
    resolve_modal
    "$MODAL_BIN" app logs "$APP_NAME"
}

show_help() {
    cat << EOF
MangaTranslator Modal phase-1

Commands:
  setup     Create secret ${SECRET_NAME} from .env.prod
  deploy    modal deploy deploy/modal_app.py
  models    Preload YOLO / OSB / LaMa / DBNet / manga-ocr / fonts onto Volume mt-models
  test      Smoke test submit + SSE (fast|precise|health)
  logs      Show app logs
  help      This message
EOF
}

case "${1:-help}" in
    setup) setup ;;
    deploy) deploy ;;
    models) download_models ;;
    test) shift; run_tests "$@" ;;
    logs) view_logs ;;
    help|--help|-h) show_help ;;
    *) print_error "Unknown command: $1"; show_help; exit 1 ;;
esac
