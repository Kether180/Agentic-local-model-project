# Build from the workspace root, not this directory:
#   docker build -f apps/pipeline/Dockerfile -t take-home-pipeline .
# The whole workspace is needed — uv.lock and every member manifest live above this app.
FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

# Install the project into `/app`
WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_DEV=1
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Resolution needs the lockfile plus the manifest of every workspace member, so bind them all in.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=apps/api/pyproject.toml,target=apps/api/pyproject.toml \
    --mount=type=bind,source=apps/pipeline/pyproject.toml,target=apps/pipeline/pyproject.toml \
    --mount=type=bind,source=packages/core/pyproject.toml,target=packages/core/pyproject.toml \
    --mount=type=bind,source=packages/domain/pyproject.toml,target=packages/domain/pyproject.toml \
    uv sync --locked --no-install-workspace --package pipeline

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --package pipeline

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT []

USER nonroot

# Run the RabbitMQ consumer by default; `pipeline` is the console script declared in
# apps/pipeline/pyproject.toml. The venv binary is called directly rather than via
# `uv run`: with UV_NO_DEV=1 a bare `uv run` would re-sync the *root* project, whose only
# dependents on `pipeline` sit in the dev group — it would uninstall this app.
# Needs a reachable broker at APP_RABBIT_URL.
CMD ["pipeline"]
