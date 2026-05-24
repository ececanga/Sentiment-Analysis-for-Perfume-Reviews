#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

PYTHON_BIN=".venv/bin/python"
CONFIG_PATH="config/gemini_judge/judge_g25pro_v1.yaml"
RUN_MODE="launch"
RUN_SMOKE_TEST=0
SMOKE_ONLY=0
SOURCE_ENV_FILE=1

usage() {
  cat <<'EOF'
Usage:
  scripts/run_phase1_judge.sh [--smoke] [--smoke-only] [--dry-run | --resume] [--no-source-env]

Behavior:
  Default      Verify environment and launch the full Phase 1 judge run
  --dry-run    Verify environment and run the branch dry-run only
  --resume     Verify environment and resume an interrupted branch run
  --smoke      Run a tiny live gemini-2.5-pro smoke call before the branch command
  --smoke-only Verify credentials with the tiny live smoke call, then exit

Notes:
  - If .env exists, the script sources it by default.
  - GEMINI_API_KEY must be available after sourcing .env or from the shell environment.
EOF
}

for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      RUN_MODE="dry-run"
      ;;
    --resume)
      RUN_MODE="resume"
      ;;
    --smoke)
      RUN_SMOKE_TEST=1
      ;;
    --smoke-only)
      RUN_SMOKE_TEST=1
      SMOKE_ONLY=1
      ;;
    --no-source-env)
      SOURCE_ENV_FILE=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${SOURCE_ENV_FILE}" -eq 1 && -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing virtualenv interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Repo: ${REPO_ROOT}"
echo -n "Python: "
"${PYTHON_BIN}" -V

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  cat >&2 <<'EOF'
GEMINI_API_KEY is not set.

Fix one of these first:
  1. export GEMINI_API_KEY='your_key_here'
  2. add GEMINI_API_KEY=your_key_here to .env

Then rerun this script.
EOF
  exit 1
fi

echo "GEMINI_API_KEY is set"

if [[ "${RUN_SMOKE_TEST}" -eq 1 ]]; then
  echo "Running live gemini-2.5-pro smoke test..."
  "${PYTHON_BIN}" - <<'PY'
from google import genai

client = genai.Client()
resp = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Reply with exactly: OK",
)
print((resp.text or "").strip())
PY
fi

if [[ "${SMOKE_ONLY}" -eq 1 ]]; then
  echo "Smoke test completed. Exiting without launching the branch run."
  exit 0
fi

COMMAND=("${PYTHON_BIN}" -m src.pipeline.run_gemini_judge_branch --config "${CONFIG_PATH}")

case "${RUN_MODE}" in
  dry-run)
    COMMAND+=(--dry-run)
    ;;
  resume)
    COMMAND+=(--resume)
    ;;
  launch)
    ;;
esac

echo "Launching command:"
printf '  %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}"
