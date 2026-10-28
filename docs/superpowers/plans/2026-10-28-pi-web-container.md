# Raspberry Pi Web Container Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal, locally accessible FastAPI container that runs on a Mac and Raspberry Pi 5 without coupling to the wildlife ML environment.

**Architecture:** Keep the service and its locked dependencies under `web/`; build that directory as the Docker context. Docker Compose publishes port 8080 only on host loopback, and the service has no persistent state or remote ingress.

**Tech Stack:** Python 3.13, FastAPI, Uvicorn, uv lockfile, Docker, Docker Compose, unittest, FastAPI TestClient.

**Spec:** `docs/superpowers/specs/2026-10-28-pi-web-container-design.md`

## Global Constraints

- Use an official Python 3.13 slim image supporting Linux ARM64 and AMD64.
- Keep web dependencies reproducible and separate from the main project's ML-heavy environment.
- Serve on container port 8080; publish only as `127.0.0.1:8080` on the host.
- Run as an unprivileged user, with one Uvicorn worker and reload disabled.
- Configure a `/health` HTTP health check, automatic restart, read-only root filesystem, dropped capabilities, and no-new-privileges.
- Do not use privileged mode, mount the Docker socket, add persistent state, or expose a non-loopback host port.
- Do not add Tailscale, tunnels, HTTPS proxies, router port forwarding, public DNS, AI HAT+ 2 integration, or application authentication.
- Keep existing wildlife tools and their dependencies unchanged.

## Review Focus

- Non-loopback publication accidentally exposes the service to LAN; test the rendered Compose port binding is exactly `127.0.0.1:8080`.
- Runtime silently runs as root or needs a writable root filesystem; inspect container user and prove startup with read-only root and only declared temporary storage.
- Build context includes repository data, models, or secrets; assert Compose uses only `web/` as build context and keep `.dockerignore` restrictive.
- ARM64 or AMD64 image fails to build due to a native-only dependency; use pure-Python web dependencies and build on each native host architecture.
- Endpoint behavior changes or becomes coupled to the main ML app; test `/`, `/health`, and unknown-route responses in an isolated web test environment.

---

### Task 1: Create The Isolated Web Application

**Files:**
- Create: `web/pyproject.toml`
- Create: `web/uv.lock`
- Create: `web/app.py`
- Test: `tests/test_web_app.py`

**Interfaces:**
- Produces `app: fastapi.FastAPI` in `web/app.py`, with `GET /` returning minimal HTML identifying wildl-id and `GET /health` returning HTTP 200 with `{"status": "ok"}`.
- The test extra provides HTTPX for `fastapi.testclient.TestClient`; production dependencies are FastAPI and Uvicorn only.
- The web project is independently lockable and does not depend on the repository's root `pyproject.toml` or `uv.lock`.

- [ ] **Step 1: Add isolated project metadata** in `web/pyproject.toml` with FastAPI and Uvicorn runtime dependencies, HTTPX in a `test` optional dependency, and Python 3.13 support. Do not depend on the repository's root project.
- [ ] **Step 2: Write tests** named `test_home_returns_wildl_id_html`, `test_health_returns_ok_json`, and `test_unknown_route_returns_404` in `tests/test_web_app.py`. Assert HTML response status/content type/title, absence of repository or device details, exact health JSON/status, and unknown path status.
- [ ] **Step 3: Run the focused tests** with `uv run --project web --extra test python -m unittest discover -s tests -p test_web_app.py`. Expected: FAIL because `web/app.py` does not exist.
- [ ] **Step 4: Add the minimal FastAPI app** in `web/app.py`; keep routes independent of wildlife modules and filesystem data.
- [ ] **Step 5: Generate and check the isolated lockfile** with `uv lock --project web`, then run `uv lock --project web --check` and `uv run --project web --locked --extra test python -m unittest discover -s tests -p test_web_app.py`. Expected: lock is current and all three tests pass.
- [ ] **Step 6: Commit** the app, tests, project metadata, and lockfile as one coherent change.

### Task 2: Build And Harden The Local Container

**Files:**
- Create: `web/Dockerfile`
- Create: `web/.dockerignore`
- Create: `compose.yaml`

**Interfaces:**
- Consumes `web/app.py` and the isolated lockfile from Task 1.
- Produces Compose service `web`, available at `http://127.0.0.1:8080` on the host.
- The container listens on `0.0.0.0:8080` internally; host publishing is restricted to `127.0.0.1`.

- [ ] **Step 1: Add a multi-platform Dockerfile** using official `python:3.13-slim`, locked runtime dependencies, and a dedicated unprivileged user. Pin any build tooling version. Start Uvicorn with one worker and reload disabled.
- [ ] **Step 2: Restrict the build context** in `web/.dockerignore` so local virtual environments, caches, and generated files are excluded. Keep Compose `build.context` exactly `./web`.
- [ ] **Step 3: Add Compose hardening**: loopback-only `127.0.0.1:8080:8080`, restart policy `unless-stopped`, health check against `/health`, read-only root, temporary writable `/tmp`, all capabilities dropped, `no-new-privileges`, no privileged mode or host mounts, and bounded local log retention.
- [ ] **Step 4: Validate Compose and build** with `docker compose config --quiet` and `docker compose build`. Expected: valid configuration and a successful native-architecture image build.
- [ ] **Step 5: Run the service** with `docker compose up -d`; verify `/` and `/health` via HTTP on `127.0.0.1:8080`, wait for Docker health status, and inspect published ports and container user. Verify the container starts with read-only root. Then run `docker compose down`.
- [ ] **Step 6: Commit** container and Compose configuration after the focused checks pass.

### Task 3: Document Use And Keep Tests In CI

**Files:**
- Create: `web/README.md`
- Modify: `.github/workflows/tests.yml`

**Interfaces:**
- Documentation commands target the root `compose.yaml` and the isolated `web` project.
- CI installs only the web test extra for Python 3.13 and runs the focused web tests without installing the ML-heavy root project for that job.

- [ ] **Step 1: Document prerequisites and commands** in `web/README.md`: 64-bit Raspberry Pi OS, Docker Engine with Compose, local start/stop/log/rebuild commands, and browsing on the host at `http://localhost:8080`. Explain that Pi loopback is reachable only from the Pi itself; no LAN or internet access is configured.
- [ ] **Step 2: Document deployment boundaries**: build on each native host architecture; Pi hardware and cross-architecture execution remain unverified; future remote access requires a separate security review and is not enabled by these instructions.
- [ ] **Step 3: Add a Python 3.13 CI step** to install the isolated web test extra, check `web/uv.lock`, and run `tests/test_web_app.py`; preserve existing CI matrix and commands.
- [ ] **Step 4: Run focused verification**: `uv lock --project web --check`, `uv run --project web --locked --extra test python -m unittest discover -s tests -p test_web_app.py`, and existing repo checks from `docs/development.md`. If Docker is available, repeat Compose validation, build, smoke checks, hardening inspection, and cleanup from Task 2.
- [ ] **Step 5: Review the final diff** for non-loopback publishing, secret/data mounts, ML dependencies, and unrelated changes. Commit docs and CI together after checks pass.

## Stop Conditions

- Stop after local native-architecture Docker verification and focused CI tests; do not configure internet access or claim Pi hardware validation.
- If Docker is unavailable, report endpoint and CI verification separately from unverified container behavior; leave no running containers.
- The design and plan documents are under globally ignored `docs/`; stage them explicitly when the owner authorises tracking.
