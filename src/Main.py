from utils.miscelaneous import turn_off_warnings
turn_off_warnings()

from utils import save_numpy, save_yaml, Multiprocess
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig):
    
    inst = cfg.instance
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    worker = Multiprocess(inst, run_dir)
    results = worker.run()

    save_numpy(str(run_dir / "results.npy"), results)

    # save_yaml(
    #     str(run_dir / "config.yaml"),
    #     pretty_yaml=OmegaConf.to_yaml(cfg.instance, sort_keys=False),
    #     raw_yaml=OmegaConf.to_container(cfg.instance, resolve=True),
    # )


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()