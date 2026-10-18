from __future__ import annotations

import sys
from pathlib import Path

# Allow running the overlay's tests before it is installed into the target repo.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
