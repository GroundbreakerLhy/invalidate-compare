#!/bin/bash
set -e
cd "$(dirname "$0")"

OUTPUT="$(dirname "$0")/llm_attack_output.txt"
DURATION_DISRUPTOR=30
DURATION_VICTIM=20

./disruptor $DURATION_DISRUPTOR &
DISRUPTOR_PID=$!

sleep 1
./delta > $OUTPUT &
DELTA_PID=$!

sleep 2
./llm_victim $DURATION_VICTIM

wait $DELTA_PID 2>/dev/null || true
kill $DISRUPTOR_PID 2>/dev/null || true
wait $DISRUPTOR_PID 2>/dev/null || true

lines=$(wc -l < "$OUTPUT")
echo "samples: $lines"
awk '{print $1}' "$OUTPUT" | sort -n | uniq -c | sort -rn
