FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY web/pyproject.toml web/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

COPY web/app.py web/visualisations.py ./web/
COPY src/labelling/review_app.py src/labelling/review_labels.py src/labelling/remove_empty_images.py src/labelling/organize_images.py src/labelling/pixel_hash.py ./src/labelling/
COPY src/labelling/review_zoom.js ./src/labelling/
COPY src/labelling/vendor/panzoom.es.js src/labelling/vendor/PANZOOM-LICENSE.txt ./src/labelling/vendor/
COPY src/visualisation/evaluate_labels.py ./src/visualisation/

FROM python:3.13-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/web /app/web
COPY --from=builder --chown=app:app /app/src/labelling /app/src/labelling
COPY --from=builder --chown=app:app /app/src/visualisation /app/src/visualisation
COPY --chown=app:app docs/assets /app/docs/assets

USER app:app

EXPOSE 8080

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
