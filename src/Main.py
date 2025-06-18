from utils import save_numpy, save_yaml, Multiprocess
import hydra
from omegaconf import DictConfig, OmegaConf
import ast
from datetime import datetime


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
  

    inst = cfg.instance


    # Run your parallel grid
    worker = Multiprocess(
        Model_selection=inst.Model_selection,
        nodes_list=[ast.literal_eval(s) for s in inst.nodes_list],
        no_inits=inst.no_inits,
        seed_value=inst.seed_value,
        lr=inst.lr,
        min_delta=inst.min_delta,
        patience=inst.patience,
        verbose=inst.verbose,
        dropout=inst.dropout,
        n_splits=inst.n_splits,
        n_process=inst.n_process,
        formulation=inst.formulation,
        cv_approach=inst.cv_approach,
        penalty=inst.penalty,
        timeout_per_node=inst.timeout_per_node,
        n_countries=inst.n_countries,
        time_periods=inst.time_periods
    )
    
    results=worker.run()


    # Save results 
    path=f"results/metrics/{inst.Model_selection}/{datetime.today().strftime('%Y-%m-%d')}/results.npy"
    save_numpy(path, results)
    
    # Save the configuration
    path = f"results/config/{inst.Model_selection}/{datetime.today().strftime('%Y-%m-%d')}/config.yaml"
    save_yaml(path, OmegaConf.to_yaml(cfg.instance, sort_keys=False))
    
    # return results
    return None

if __name__ == "__main__":    main()



