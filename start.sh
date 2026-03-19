#!/bin/bash
# start.sh - Launch mulmodel auto-research
#
# Usage:
#   ./start.sh              # Auto mode (default)
#   ./start.sh --inspect    # Inspect mode (clearer output)
#   ./start.sh --continuous # Keep running (auto-restart on failure)
#   ./start.sh --serve      # Use persistent server mode (recommended for stability)
#   ./start.sh --timeout 900 # Set timeout to 900 seconds
#   ./start.sh --help        # Show help
#
# Examples:
#   ./start.sh                              # Run once
#   ./start.sh --continuous                 # Run forever, auto-restart on crash
#   ./start.sh --continuous --serve         # Most stable mode for long sessions
#   ./start.sh --continuous --timeout 900   # 15 min timeout, keep running
#
# Connection stability tips:
#   - Use --serve for persistent server (prevents disconnection)
#   - Use --continuous to auto-recover from crashes
#   - Each run is independent — no state leaks between calls
#   - If "unable to connect": try 'kilocode auth login' first

set -e

cd "$(dirname "$0")"

# Defaults
INSPECT=""
CONTINUOUS=""
SERVE=""
TIMEOUT="600"
HELP=""

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --inspect)   INSPECT="--inspect"; shift ;;
        --continuous) CONTINUOUS="--continuous"; shift ;;
        --serve)     SERVE="--serve"; shift ;;
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        --help|-h)
            HELP="yes"; shift ;;
        *) echo "Unknown flag: $1"; echo "Run with --help for usage."; exit 1 ;;
    esac
done

if [[ "$HELP" == "yes" ]]; then
    echo "Usage: $0 [flags]"
    echo ""
    echo "Flags:"
    echo "  --inspect      Inspect mode (clearer output)"
    echo "  --continuous   Keep running (auto-restart on failure)"
    echo "  --serve        Use persistent server mode (recommended for stability)"
    echo "  --timeout N    Timeout in seconds (default 600)"
    echo "  --help         Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --continuous --serve  # Most stable: persistent server + auto-restart"
    exit 0
fi

# Build command
PARTS=()
[[ -n "$INSPECT" ]]    && PARTS+=("$INSPECT")
[[ -n "$CONTINUOUS" ]] && PARTS+=("$CONTINUOUS")
[[ -n "$SERVE" ]]      && PARTS+=("$SERVE")
PARTS+=("--timeout" "$TIMEOUT")

CMD="python3 start.py ${PARTS[*]}"

echo "[start.sh] Working dir: $(pwd)"
echo "[start.sh] Command: $CMD"
echo ""
exec $CMD
