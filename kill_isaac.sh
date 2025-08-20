#!/usr/bin/env bash
# kill_isaac.sh — one-click killer for Isaac Sim / Omniverse Kit processes

set -euo pipefail

LIST_ONLY=false
ONLY_MINE=false
FORCE_KILL=false

for arg in "$@"; do
  case "$arg" in
    --list) LIST_ONLY=true ;;
    --me) ONLY_MINE=true ;;
    --force) FORCE_KILL=true ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# 匹配关键词
REGEX='isaac-sim|Omniverse/Isaac-Sim|omni.kit|kit/python/bin/python3'

# 拿到进程列表
if $ONLY_MINE; then
  PROCS=$(ps -u "$(id -u)" -o pid=,args= | grep -E "$REGEX" | grep -v grep || true)
else
  PROCS=$(ps -eo pid=,args= | grep -E "$REGEX" | grep -v grep || true)
fi

if [ -z "$PROCS" ]; then
  echo "No Isaac/Kit processes matched."
  exit 0
fi

echo "Matched processes:"
echo "$PROCS"

PIDS=$(echo "$PROCS" | awk '{print $1}')

$LIST_ONLY && exit 0

if $FORCE_KILL; then
  echo "Force killing (SIGKILL)..."
  kill -9 $PIDS || true
  exit 0
fi

echo "Sending SIGTERM..."
kill $PIDS || true
sleep 3

# 再查一遍，剩下的直接 KILL -9
if $ONLY_MINE; then
  LEFT=$(ps -u "$(id -u)" -o pid=,args= | grep -E "$REGEX" | grep -v grep | awk '{print $1}' || true)
else
  LEFT=$(ps -eo pid=,args= | grep -E "$REGEX" | grep -v grep | awk '{print $1}' || true)
fi

if [ -n "$LEFT" ]; then
  echo "Still running, sending SIGKILL..."
  kill -9 $LEFT || true
fi

echo "Done."