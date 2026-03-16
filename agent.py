"""
agent.py — Luminar Subnet Test Agent
-------------------------------------
Minimal agent for testing the validator pipeline end-to-end.

Setup phase: downloads a tiny file from HuggingFace to verify
             network access and /cache/ write access work.

Infer phase: writes hardcoded predictions to output.csv —
             no model loading, no GPU needed.

Keep all your agent logic between START and END.

"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

OUTPUT_DIR  = Path("/data/output")
OUTPUT_FILE = OUTPUT_DIR / "output.csv"
CACHE_DIR   = Path("/cache")

############################# START #############################

##AGENT LOGIC AND CODE HERE

############################# END #############################


def setup() -> None:
    """
    Download a tiny file from HuggingFace to verify:
      - Network access is available in setup phase
      - /cache/ is writable
    """
    # setup phase


def infer() -> None:
    """Write hardcoded predictions to output.csv."""

    # infer phase


if __name__ == "__main__":
    if "--setup" in sys.argv:
        setup()
    elif "--infer" in sys.argv:
        infer()
    else:
        print("Usage: python agent.py --setup | --infer")
        sys.exit(1)
