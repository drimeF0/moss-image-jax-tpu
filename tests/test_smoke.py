import os
import subprocess
import sys
import json
from pathlib import Path

import numpy as np

from mossjax import init, smoke, text_encoder as jte, vae as jv, weights


def test_init_tree_rules():
    tree = init.init_tree(weights.param_shapes(smoke.SMOKE_DIT), __import__("jax").random.PRNGKey(0), ("layers", "blocks"))
    assert float(np.asarray(tree["layers"][0]["norm1"]["weight"]).min()) == 1.0
    assert float(np.asarray(tree["video_embedder"]["bias"]).max()) == 0.0
    te_tree = init.init_tree(jte.param_shapes(smoke.SMOKE_TE), __import__("jax").random.PRNGKey(1), ("layers",))
    assert float(np.asarray(te_tree["layers"][0]["linear_attn"]["dt_bias"]).min()) == 1.0
    vae_tree = init.init_tree(jv.param_shapes(smoke.SMOKE_VAE), __import__("jax").random.PRNGKey(2), jv.VAE_LIST_CONTAINERS)
    assert float(np.asarray(vae_tree["bn"]["running_var"]).min()) == 1.0


def test_colab_runner_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "colab_run.py"), "--mode", "smoke", "--steps", "2", "--sample-steps", "2", "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(root),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["stages"]["smoke"]["status"] == "ok"
    assert (tmp_path / "smoke_sample.png").exists()
