# Testing Environment Design: Data–Detector Alignment

## Executive Summary

The detection mechanisms are trained on **benchmark datasets** (UNSW-NB15, CICIDS2017) normalized to a canonical 17-feature flow schema. The **original HTTP-based testing environment** produces application-level metadata (path length, headers, User-Agent) with value distributions that differ sharply from benchmark data. Models trained on UNSW/CICIDS see synthetic HTTP flows as **out-of-distribution**, making the HTTP-based env unsuitable for validating those detectors.

This document describes the mismatch, related frameworks, and the **two-mode testing architecture** adopted in this project.

---

## 1. The Data–Detector Mismatch

### 1.1 Benchmark Data (Training Source)

| Canonical field | UNSW-NB15 source | Value range (typical) |
|-----------------|------------------|------------------------|
| path_length | smean (mean source packet size) | ~45–800 bytes |
| query_length | dmean (mean dest packet size) | ~0–100 bytes |
| duration_ms | dur (flow duration, sec→ms) | 0.001–30 000 ms |
| header_count | sttl (source TTL) | 0–255 |
| header_size_bytes | dttl (dest TTL) | 0–255 |
| inter_arrival_ms | sinpkt (inter-packet time) | 0.01–10 000 ms |
| request_sequence | spkts (source packets) | 1–50 |
| requests_last_60s | dpkts (dest packets) | 0–50 |
| unique_paths_count | ct_srv_src | 1–10+ |
| hour_utc, minute | ct_src_ltm, ct_dst_ltm | 0–100+ (activity proxies) |
| user_agent_length, referer_present | (no match) | 0 |

Benchmark flows are **network-level** statistics (packets, bytes, TTL, inter-packet times). The canonical schema maps these to names like `path_length` for API consistency; the semantics remain packet/flow oriented.

### 1.2 HTTP Synthetic Data (Original Testing Env)

| Canonical field | Synthetic HTTP source | Value range (typical) |
|-----------------|------------------------|------------------------|
| path_length | Actual URL path length (chars) | 1–80 |
| query_length | Query string length (chars) | 0–40 |
| duration_ms | Request handling time | 1–500 ms |
| header_count | Number of HTTP headers | 5–20 |
| header_size_bytes | Total header size (bytes) | 200–800 |
| user_agent_length | User-Agent string length | 30–80 |
| referer_present | 0/1 Referer header | 0 or 1 |
| inter_arrival_ms | Time since last request | 2000–10 000 ms (2–10 s interval) |
| request_sequence | Per-client request count | 1–100+ |
| unique_paths_count | Distinct paths per client | 1–5 |

Here the same column names are filled with **application-level** HTTP semantics. Value ranges and correlations differ from the benchmark data. A detector trained on UNSW will treat synthetic HTTP flows as OOD and produce arbitrary or misleading results.

### 1.3 Conclusion

- **HTTP synthetic env** is unsuitable for validating models trained on UNSW-NB15 or CICIDS2017.
- HTTP env remains useful for:
  - Testing the pipeline (receiver → flows.csv → detector) end-to-end.
  - Future HTTP-specific models trained on HTTP traffic.
- For detector validation on benchmark-trained models, the testing env must use **benchmark-derived flows** (in-distribution data).

---

## 2. Related Frameworks

| Framework | Purpose | Relevance |
|-----------|---------|------------|
| **FLAME** (ETH Zurich) | Inject anomalies into NetFlow/IPFIX traces | Flow-level; requires NetFlow format. Our canonical schema differs. |
| **ISCXFlowMeter** | Flow generator/analyzer for ISCX datasets | Flow-oriented; different schema. |
| **ID2T** | Inject synthetic attacks into background traffic | Packet-level; heavy integration. |
| **NAD-Eval** (NIST) | Evaluate NAD systems with synthetic data | High-level; limited documentation. |

These focus on NetFlow, packet capture, or custom schemas. Our setup uses a custom canonical 17-feature schema. The most direct approach is to **replay flows from our own normalized benchmarks** so the detector sees the same feature space it was trained on.

---

## 3. Two-Mode Architecture

### 3.1 Mode A: Benchmark Replay (Recommended for Validation)

**Purpose**: Validate detectors trained on UNSW-NB15 / CICIDS2017.

**Flow**:
```
canonical_train.csv / canonical_val.csv  →  flow-replayer  →  flows.csv  →  detector
```

The **flow replayer** streams rows from prepared canonical CSVs into `flows.csv`, with optional inter-row delay to simulate live traffic. Data is **identical** to training/validation data: same features, same distributions.

**Use cases**:
- Live daemon demo with in-distribution flows.
- Integration tests for the detector pipeline.
- Reproducible evaluation without HTTP containers.

### 3.2 Mode B: HTTP Synthetic (Pipeline / HTTP-Specific)

**Purpose**: Exercise the full HTTP pipeline; future HTTP-trained models.

**Flow**:
```
HTTP generator  →  receiver  →  flows.csv  →  detector
```

The generator sends HTTP requests; the receiver logs them as flows. Anomalies are simulated (suspicious paths, User-Agents). Data is **not** compatible with benchmark-trained models.

**Use cases**:
- Smoke-testing receiver and detector wiring.
- HTTP-specific training and testing (when implemented).

---

## 4. Implementation Summary

| Component | Location | Role |
|-----------|----------|------|
| Flow replayer | `testing-environment/flow-replayer/` | Read canonical CSV; stream rows to flows.csv |
| Docker Compose (benchmark) | `docker-compose.benchmark.yml` | flow-replayer + detector; no receiver/generator |
| Demo (benchmark) | `demo-benchmark.sh` | Start benchmark env; run daemon on replayed flows |
| Demo (HTTP) | `demo.sh` | Original: receiver + generator + daemon |

---

## 5. When to Use Which Mode

| Goal | Use |
|------|-----|
| Validate detector trained on UNSW/CICIDS | **Benchmark replay** (`demo-benchmark.sh`, `make env-benchmark`) |
| Test that receiver + detector work end-to-end | **HTTP synthetic** (`demo.sh`, `make env-up`) |
| Train on HTTP traffic (future) | HTTP synthetic (collect flows, train, then use HTTP env for eval) |
| Reproducible thesis experiments | Benchmark datasets via `make evaluate-all` (no Docker) |
