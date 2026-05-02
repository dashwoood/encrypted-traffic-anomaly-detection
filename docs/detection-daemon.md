# Detection Daemon

The detection mechanism can run as a **daemon** on a system, continuously monitoring a flows CSV file and reporting anomalies. It requires read access to the flows file and optional write access for model persistence and output logs.

## Requirements

- **Files**: Read access to `flows.csv` (path from receiver or any compatible flow export)
- **Traffic source**: The flows file is produced by the receiver (see [testing-environment.md](testing-environment.md)) or any exporter that writes the same CSV schema
- **Permissions**: The daemon user must be able to read the flows file; if using `--model-dir` and `--output`, write access to those paths

## Installation

### From source (development)

```bash
cd detection-mechanisms
pip install -e .
```

### From wheel (packaged)

```bash
pip install dist/flow_anomaly_detector-*.whl
```

### From PyPI (after release)

```bash
pip install flow-anomaly-detector
```

## CLI Commands

| Command       | Description                                      |
|---------------|--------------------------------------------------|
| `detect track`  | Batch mode: process flows once, print anomalies   |
| `detect daemon` | Daemon mode: poll flows file, report continuously |
| `detect list-models` | List available models                     |

## Daemon Usage

### Basic

```bash
detect daemon --flows /path/to/flows.csv --model isolation_forest
```

### With model persistence

```bash
detect daemon \
  --flows /var/log/detector/flows.csv \
  --model isolation_forest \
  --model-dir /var/lib/detector \
  --interval 5
```

- **`--flows`** (required): Path to the flows CSV file
- **`--model`**: Model name (default: `isolation_forest`). Use `detect list-models` to see options
- **`--interval`**: Poll interval in seconds (default: 5)
- **`--model-dir`**: Directory to save/load the trained model; enables persistence across restarts
- **`--output`**: Write anomaly log to file (default: stdout)
- **`--min-fit-flows`**: Minimum flows to collect before fitting (default: 10)
- **`--log-level`**: `DEBUG`, `INFO`, `WARNING`, `ERROR`

### With file output

```bash
detect daemon \
  --flows /var/log/detector/flows.csv \
  --model isolation_forest \
  --output /var/log/detector/anomalies.log \
  --model-dir /var/lib/detector
```

## Running as a System Service (systemd)

### 1. Create a dedicated user (optional, recommended)

```bash
sudo useradd -r -s /bin/false detector
```

### 2. Create directories and set permissions

```bash
sudo mkdir -p /var/log/detector
sudo mkdir -p /var/lib/detector
sudo chown detector:detector /var/log/detector /var/lib/detector
```

Ensure the flows file is readable by the detector user. If the receiver writes to `/app/logs/flows.csv` inside Docker, mount that path and give the detector user read access (e.g. via group or world-readable).

### 3. Install the package

```bash
sudo pip install flow-anomaly-detector
# Or: pip install --user and use the user's bin path
```

### 4. Create systemd unit

Create `/etc/systemd/system/detector.service`:

```ini
[Unit]
Description=Flow Anomaly Detection Daemon
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=detector
Group=detector
ExecStart=/usr/local/bin/detect daemon \
  --flows /var/log/detector/flows.csv \
  --model isolation_forest \
  --model-dir /var/lib/detector \
  --output /var/log/detector/anomalies.log \
  --interval 5
Restart=on-failure
RestartSec=10

# Security
NoNewPrivileges=yes
PrivateTmp=yes
ReadOnlyPaths=/
ReadWritePaths=/var/log/detector /var/lib/detector

[Install]
WantedBy=multi-user.target
```

Adjust paths to match your setup. If the flows file is in a Docker volume, ensure it is mounted to a path the service can read (e.g. `/var/log/detector/flows.csv`).

If using `ReadOnlyPaths=/`, remove it or add `ReadWritePaths` for any path the process needs to write.

### 5. Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable detector
sudo systemctl start detector
sudo systemctl status detector
```

### 6. View logs

```bash
# Service logs
sudo journalctl -u detector -f

# Anomaly output
tail -f /var/log/detector/anomalies.log
```

## Docker Integration

Run the daemon alongside the receiver by mounting the flows volume:

```yaml
# In docker-compose.yml (add service)
  detector:
    image: python:3.11-slim
    working_dir: /app
    command: >
      sh -c "pip install flow-anomaly-detector &&
             detect daemon --flows /data/flows.csv --model isolation_forest"
    volumes:
      - ./data/logs:/data
    depends_on:
      - receiver
    restart: unless-stopped
```

Or build a custom image that includes the detector and your flows path.

## File Access Summary

| Path              | Access | Purpose                          |
|-------------------|--------|----------------------------------|
| `--flows`         | Read   | Flow data from receiver/export   |
| `--model-dir`     | Read+Write | Persist trained model         |
| `--output`        | Write  | Anomaly log output               |

## Troubleshooting

- **Permission denied** on flows file: Ensure the daemon user (or current user) can read the CSV. If the receiver runs in Docker, check volume mounts and file ownership.
- **No flows to process**: The file may be empty or not yet created. The daemon will wait and retry each interval.
- **Model not fitting**: Ensure at least `--min-fit-flows` (default 10) rows exist in the CSV before the first detection run.
