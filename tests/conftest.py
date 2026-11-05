import sys
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "reference" / "kohya_sd_scripts"))

jax.config.update("jax_enable_x64", True)
