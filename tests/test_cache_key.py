"""The cache key is the whole point of the pipeline — it decides what gets recomputed."""

from core.hashing import cache_key, config_hash, sha256_bytes

SHA = sha256_bytes(b"document bytes")


def key(sha: str = SHA, step: str = "embed", version: int = 1, cfg: str = "") -> str:
    return cache_key(sha, step, version, cfg or config_hash({"model": "nomic"}))


def test_same_inputs_same_key():
    assert key() == key()


def test_changed_content_busts_the_cache():
    assert key() != key(sha=sha256_bytes(b"different bytes"))


def test_step_version_busts_the_cache():
    assert key() != key(version=2)


def test_step_config_busts_the_cache():
    assert key() != key(cfg=config_hash({"model": "mxbai"}))


def test_steps_do_not_collide():
    assert key(step="embed") != key(step="chunk")


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
