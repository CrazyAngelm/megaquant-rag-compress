from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'compare_hdc_turbovec_rotor.py'


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    sys.argv[0] = str(SCRIPT)
    runpy.run_path(str(SCRIPT), run_name='__main__')
