#!/usr/bin/env bash
# Run the training queue in the background using nohup.
# Output is saved to queue_output.log in this directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/queue_output.log"
PYTHON="$SCRIPT_DIR/../.venv/bin/python3"
QUEUE_SCRIPT="$SCRIPT_DIR/run_queue.py"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python venv not found at $PYTHON"
    exit 1
fi

if [ ! -f "$QUEUE_SCRIPT" ]; then
    echo "ERROR: run_queue.py not found at $QUEUE_SCRIPT"
    exit 1
fi

cd "$SCRIPT_DIR"

echo "Starting training queue in the background..."
echo "  Log file : $LOG_FILE"
echo "  Python   : $PYTHON"

nohup "$PYTHON" "$QUEUE_SCRIPT" > "$LOG_FILE" 2>&1 &
PID=$!

echo "  PID      : $PID"
echo ""
echo "Queue is running. To monitor progress:"
echo "  tail -f $LOG_FILE"
echo "  tail -f $SCRIPT_DIR/queue_progress.log"
echo ""
echo "To stop the queue:"
echo "  kill $PID"
