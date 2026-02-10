from utils.miscelaneous import turn_off_warnings
turn_off_warnings()

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
    path=f"results/metrics/{inst.formulation}/{inst.model_selection}/{datetime.today().strftime('%Y-%m-%d')}/results.npy"
    save_numpy(path, results)
    
    # Save the configuration
     
    path = f"results/config/{inst.formulation}/{inst.model_selection}/{datetime.today().strftime('%Y-%m-%d')}/config.yaml"
    if inst.formulation == 'regional':
        #remove the cfg.instance.model_selection attribute string from path 
        path = path.replace(f"/{inst.model_selection}/", "/")
    save_yaml(path, pretty_yaml=OmegaConf.to_yaml(cfg.instance, sort_keys=False), raw_yaml=OmegaConf.to_container(cfg.instance, resolve=True))
    
    # return results
    return None

if __name__ == "__main__":    
    
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    main()

