FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

# Security best practices
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first to leverage Docker cache for dependency install
COPY pyproject.toml uv.lock /app/

# Install dependencies _before_ adding app source for optimal Docker caching,
# ensures dependencies are only reinstalled when the lock file or pyproject changes
RUN uv sync --frozen

# Now copy the application source code (changes here won't invalidate dependency layer)
COPY fastapi_app /app/fastapi_app

FROM python:3.11-slim AS runtime

# Install uv runtime only (no build tools/gcc for minimum attack surface)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

WORKDIR /app

# Copy everything from /app, including site-packages, to runtime image
COPY --from=builder /app /app

# Ensure directories for non-root execution and logging exist
RUN mkdir -p /app/logs && \
    adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

# Use non-root user
USER appuser

EXPOSE 8000

# Use an unprivileged, explicit command
CMD ["uv", "run", "gunicorn", "fastapi_app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "--threads", "2"]

