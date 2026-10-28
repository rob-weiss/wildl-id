# Local Web Container

This minimal FastAPI service is a separate starting point for the wildl-id web
application. It runs in Docker and does not load the project's images, models,
labels, or machine-learning dependencies.

## Requirements

- macOS with Docker Desktop, or another system with Docker Engine and Compose.
- Raspberry Pi 5 with 64-bit Raspberry Pi OS and Docker Engine with Compose.

The image builds for the native architecture of the Docker host. Build it on
each host rather than copying an architecture-specific image between machines.

## Run Locally

From the repository root, build and start the service:

```bash
docker compose up --build -d
```

Open <http://localhost:8080>. Check the health endpoint at
<http://localhost:8080/health>. View logs with:

```bash
docker compose logs -f web
```

Rebuild after changing the app or its dependencies:

```bash
docker compose up --build -d
```

## Update

From the repository root, pull the latest source and rebuild the service:

```bash
git pull --ff-only
docker compose up --build -d
```

Stop and remove the container with:

```bash
docker compose down
```

The service has no persistent state. On the Pi, `localhost` refers to the Pi,
so open the page in a browser running on the Pi itself. The container publishes
port 8080 only on `127.0.0.1`; it is not reachable from the LAN or internet.

No remote ingress or authentication is configured. Remote access, invited
users, or public access need a separate security review and configuration.
Pi hardware and cross-architecture execution have not been verified.
