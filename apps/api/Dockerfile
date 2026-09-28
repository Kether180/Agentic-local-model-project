# Build from the workspace root, not this directory:
#   docker build -f apps/api/Dockerfile -t take-home-api .
# The whole workspace is needed — uv.lock and every member manifest live above this app.
FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim

RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

# Install the project into `/app`
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_DEV=1
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Resolution needs the lockfile plus the manifest of every workspace member,
# so bind them all in. Source still comes later, for optimal layer caching.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=apps/api/pyproject.toml,target=apps/api/pyproject.toml \
    --mount=type=bind,source=apps/pipeline/pyproject.toml,target=apps/pipeline/pyproject.toml \
    --mount=type=bind,source=packages/core/pyproject.toml,target=packages/core/pyproject.toml \
    --mount=type=bind,source=packages/domain/pyproject.toml,target=packages/domain/pyproject.toml \
    uv sync --locked --no-install-workspace --package api

# Then, add the rest of the project source code and install it
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --package api

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Use the non-root user to run our application
USER nonroot

EXPOSE 8000

# Run the FastAPI application by default.
CMD ["fastapi", "dev", "apps/api/src/api", "--host", "0.0.0.0"]
