"""Content hashing and pipeline cache keys.

The cache key is what makes re-importing a document cheap, so what goes into it decides what a
re-import is allowed to skip.
"""

import hashlib
import json
from collections.abc import Mapping
from typing import TypeAlias

ConfigValue: TypeAlias = str | int | float | bool
StepConfig: TypeAlias = Mapping[str, ConfigValue]
"""The settings a step's output depends on — model id, chunk size, and the like."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_hash(config: StepConfig) -> str:
    """Stable hash of a step's config. `sort_keys` so key order never changes the hash."""
    return hashlib.sha256(json.dumps(dict(config), sort_keys=True).encode()).hexdigest()[:16]


def cache_key(document_sha: str, step_name: str, step_version: int, cfg_hash: str) -> str:
    """Identity of one step's output.

    All four parts matter, and each closes a different way of serving a stale result:

    - `document_sha` — the document's *content*, not its id. Keying on an id means the same id with
      changed bytes replays an artifact built from the old ones.
    - `step_name` — so two steps with identical config never collide.
    - `step_version` — lets a change in a step's code invalidate just that step.
    - `cfg_hash` — catches a model swap or a chunk-size change.
    """
    return hashlib.sha256(
        f"{document_sha}|{step_name}|v{step_version}|{cfg_hash}".encode()
    ).hexdigest()
