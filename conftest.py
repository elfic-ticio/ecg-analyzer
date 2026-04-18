import sys
from pathlib import Path

# Make `src.*` importable from any working directory when running pytest
sys.path.insert(0, str(Path(__file__).parent))
