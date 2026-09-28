import logging
import os


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=level or os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )
