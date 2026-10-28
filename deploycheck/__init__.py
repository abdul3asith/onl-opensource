"""deploycheck: profile, optimize and verify robot policies on real hardware."""

from pathlib import Path

__version__ = "0.1.0"

# Repo root, derived from this file's location: works on any machine.
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
