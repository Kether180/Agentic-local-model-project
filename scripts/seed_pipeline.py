"""Push a directory of PDFs through the pipeline and report what happened.

    uv run python scripts/seed_pipeline.py samples

Uploads every PDF, waits for each to finish, prints a per-file table, and exits non-zero if any
document failed or timed out. Run it twice: the second run should show cache hits everywhere,
which is the whole point of the step cache.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

EXIT_FAILURES = 1
EXIT_NO_INPUT = 2

TERMINAL = {"completed", "failed"}


@dataclass
class Result:
    """One document's journey, as the report renders it."""

    path: Path
    document_id: str | None = None
    status: str = "pending"
    pages: int | None = None
    chunks: int = 0
    cache_hits: int = 0
    steps: int = 0
    stage: str = ""
    elapsed: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    @property
    def cached(self) -> str:
        return f"{self.cache_hits}/{self.steps}" if self.steps else "-"


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def failures(self) -> list[Result]:
        return [r for r in self.results if not r.ok]


def find_pdfs(directory: Path, pattern: str, recursive: bool) -> list[Path]:
    globber = directory.rglob if recursive else directory.glob
    return sorted(p for p in globber(pattern) if p.is_file())


def upload(client: httpx.Client, path: Path) -> str:
    """POST one PDF. Returns immediately with 202 — the work happens on the queues."""
    with path.open("rb") as handle:
        response = client.post("/documents", files={"file": (path.name, handle, "application/pdf")})
    response.raise_for_status()
    return str(response.json()["document_id"])


def refresh(client: httpx.Client, result: Result) -> None:
    """Read current state into `result`. Leaves status alone if the document is still working."""
    if result.document_id is None:
        return
    response = client.get(f"/documents/{result.document_id}")
    response.raise_for_status()
    body = response.json()

    steps = body.get("steps") or []
    result.pages = body.get("page_count")
    result.chunks = body.get("chunk_count") or 0
    result.steps = len(steps)
    result.cache_hits = sum(1 for s in steps if s.get("cache_hit"))
    if steps:
        result.stage = f"{steps[-1].get('step_name')} {steps[-1].get('step_status')}"

    failed = [s for s in steps if s.get("step_status") == "failed"]
    if failed:
        result.status = "failed"
        result.error = failed[0].get("error_msg") or f"{failed[0].get('step_name')} failed"
    else:
        result.status = str(body.get("status", "pending"))


def run(
    directory: Path, api: str, pattern: str, recursive: bool, timeout: float, interval: float
) -> Report:
    pdfs = find_pdfs(directory, pattern, recursive)
    if not pdfs:
        raise FileNotFoundError(f"no files matching {pattern!r} in {directory}")

    report = Report(results=[Result(path=p) for p in pdfs])
    started = time.monotonic()

    with httpx.Client(base_url=api, timeout=120.0) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError as err:
            raise ConnectionError(f"cannot reach {api}: {err}") from err

        print(f"uploading {len(pdfs)} pdf(s) to {api}")
        for result in report.results:
            try:
                result.document_id = upload(client, result.path)
            except (httpx.HTTPError, KeyError) as err:
                result.status = "failed"
                result.error = f"upload failed: {err}"

        deadline = time.monotonic() + timeout
        pending = [r for r in report.results if r.document_id and r.status not in TERMINAL]
        while pending and time.monotonic() < deadline:
            time.sleep(interval)
            for result in list(pending):
                stage = result.stage
                try:
                    refresh(client, result)
                except httpx.HTTPError as err:
                    result.status = "failed"
                    result.error = f"status check failed: {err}"
                if result.stage != stage:
                    print(f"  {result.path.name}: {result.stage}", flush=True)
                if result.status in TERMINAL:
                    result.elapsed = time.monotonic() - started
                    pending.remove(result)

        for result in pending:
            result.status = "timeout"
            result.error = f"still running after {timeout:.0f}s"
            result.elapsed = time.monotonic() - started

    report.elapsed = time.monotonic() - started
    return report


def render(report: Report) -> None:
    width = max([len(r.path.name) for r in report.results] + [len("file")])
    print(
        f"\n{'file':<{width}}  {'status':<10} {'pages':>5} {'chunks':>7} {'cached':>7} {'time':>7}"
    )
    for r in report.results:
        pages = "-" if r.pages is None else str(r.pages)
        chunks = str(r.chunks) if r.chunks else "-"
        print(
            f"{r.path.name:<{width}}  {r.status:<10} {pages:>5} {chunks:>7} "
            f"{r.cached:>7} {r.elapsed:>6.1f}s"
        )

    ok = len(report.results) - len(report.failures)
    print(f"\n{ok}/{len(report.results)} completed in {report.elapsed:.1f}s")
    sys.stdout.flush()
    for r in report.failures:
        print(f"  {r.path.name}: {r.error}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="seed_pipeline",
        description="Push a directory of PDFs through the pipeline and report the outcome.",
        epilog="Run it twice — the second run should be all cache hits.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="samples",
        type=Path,
        help="directory of PDFs (default: samples)",
    )
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--pattern", default="*.pdf", help="glob pattern (default: *.pdf)")
    parser.add_argument("--recursive", action="store_true", help="descend into subdirectories")
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="seconds to wait for all documents"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        dest="poll_interval",
        help="seconds between status checks",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(
            directory=args.directory,
            api=args.api.rstrip("/"),
            pattern=args.pattern,
            recursive=args.recursive,
            timeout=args.timeout,
            interval=args.poll_interval,
        )
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return EXIT_NO_INPUT
    except ConnectionError as err:
        print(f"{err}\nis the stack up? (docker compose up -d)", file=sys.stderr)
        return EXIT_FAILURES

    render(report)
    return EXIT_FAILURES if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
