import os
import multiprocessing as mp

# Ensure we use spawn 
try:
    mp.set_start_method("spawn", force=True)
    print("Multiprocessing start method set to 'spawn'")
except RuntimeError:
    # If already set, ignore
    pass


os.environ["CUDA_VISIBLE_DEVICES"] = ""     # comment this out if you need GPUs
# quiet TF logging (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from utils import save_numpy, save_yaml, Multiprocess
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
  
    inst=cfg.instance 
    
    # Run parrallel processing passing the configuration
    worker = Multiprocess(inst)
    results=worker.run()

    # Save results 
    path=f"results/metrics/{inst.model_selection}/{datetime.today().strftime('%Y-%m-%d')}/results.npy"
    save_numpy(path, results)
    
    # Save the configuration
    path = f"results/config/{inst.model_selection}/{datetime.today().strftime('%Y-%m-%d')}/config.yaml"
    save_yaml(path, OmegaConf.to_yaml(cfg.instance, sort_keys=False))
    
    # return results
    return None

if __name__ == "__main__":    
    
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    
    main()

