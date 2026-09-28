"""Worker entrypoint: `pipeline <step>`.

One image, one service per step — `command: [pipeline, embed]` in compose. Scaling a step is
`--scale pipeline-embed=4`, which is the whole reason each step gets its own queue: the expensive
ones scale without touching the rest.
"""

import argparse

from core.logging import setup_logging
from domain import StepName

from pipeline.steps import STEPS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one pipeline step worker.")
    parser.add_argument("step", choices=[s.value for s in StepName], help="which step to run")
    args = parser.parse_args()

    setup_logging()
    STEPS[StepName(args.step)]().run()
