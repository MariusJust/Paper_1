#!/usr/bin/env python3
import os

# ─── 1) Set TF/XLA/CUDA env vars before anything else ───────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""

# ─── 2) Make sure your src/ is on PYTHONPATH, if needed ───────────────────
# If your project root is the parent of src/, uncomment these lines:
# ROOT = os.path.dirname(__file__)
# sys.path.insert(0, os.path.join(ROOT, "src"))

# ─── 3) Now import & invoke your warnings‑silencer ───────────────────────
# This must happen before TF or Keras ever loads, so it can patch absl & progbar.
from utils.miscelaneous import turn_off_warnings
turn_off_warnings()

# ─── 4) Only now import Hydra & your Monte Carlo entrypoint ──────────────
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../../config", config_name="config_mc")
def run_mc(cfg: DictConfig):
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # defer all TF imports until after your patch
    from simulations.monte_carlo import mc_loop

    specs = cfg.mc.specifications
    for spec in specs:
        print(f"\n=== Running Monte Carlo for specification: {spec} ===")
        print("\n=== Running initial training loop ===")
        mc_loop(cfg, spec)
    print("\nAll specifications processed.")

if __name__ == "__main__":
    run_mc()
