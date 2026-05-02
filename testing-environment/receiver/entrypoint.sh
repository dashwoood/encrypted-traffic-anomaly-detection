#!/bin/sh
# Start tcpdump for packet capture, then the HTTP server.
# Packet-level capture supplements flow-level tracking (both used in anomaly detection research).

CAPTURE_PATH="${CAPTURE_PATH:-/app/logs/capture.pcap}"
if command -v tcpdump >/dev/null 2>&1; then
  tcpdump -i any -w "$CAPTURE_PATH" port 8000 -U 2>/dev/null &
fi
exec python server.py
