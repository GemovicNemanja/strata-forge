# syntax=docker/dockerfile:1.7

# --- builder stage --------------------------------------------------
# Compile-time toolchain lives here; the runtime image discards it.
FROM python:3.14-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Pin uv to a release tag so builds are reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first (layer caches when only source changes).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Then the source.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# --- runtime stage --------------------------------------------------
# Minimal layer: just CA certs + the prebuilt venv + non-root user.
FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1000 forge \
    && useradd --system --uid 1000 --gid forge --create-home --home-dir /home/forge forge

WORKDIR /app
COPY --from=builder --chown=forge:forge /app /app

USER forge

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD forge --help >/dev/null 2>&1 || exit 1

ENTRYPOINT ["forge"]
CMD ["--help"]
