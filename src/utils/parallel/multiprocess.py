from multiprocessing import Pool
from tqdm import tqdm
from multiprocessing import TimeoutError
from .builders import build_arg_list_cv, build_arg_list_ic
import ast


class Multiprocess:
    
    """
    This class is designed to run either cross-validation (CV) or information criteria (IC) based model selection in parallel.
    It initializes with configuration parameters (from the config folder) and data, builds the argument list for each node, and executes the training in parallel.
    The results are stored in a dictionary where keys are node indices and values are lists containing either cross-validation errors, BIC/AIC values or holdout errors.
    """
    def __init__(self, cfg, data=None):
        self.Model_selection = cfg.model_selection
        self.nodes_list = [ast.literal_eval(s) for s in cfg.nodes_list]
        # self.no_inits = cfg.no_inits
        # self.seed_value = cfg.seed_value
        # self.lr = cfg.lr
        # self.min_delta = cfg.min_delta
        # self.patience = cfg.patience
        # self.verbose = cfg.verbose
        # self.dropout = cfg.dropout
        # self.n_splits = cfg.n_splits
        # self.n_process = cfg.n_process
        # self.cv_approach = cfg.cv_approach
        # self.penalty = cfg.penalty
        # self.timeout_per_node = cfg.timeout_per_node
        # self.n_countries = cfg.n_countries
        # self.time_periods = cfg.time_periods
        
        # self.country_trends = cfg.country_trends
        # self.dynamic_model = cfg.dynamic_model
        # self.holdout=cfg.holdout
        # self.within_transform=cfg.within_transform
        # self.formulation=cfg.formulation
        self.cfg=cfg
        self.data = data

    def run(self):
        # if self.Model_selection == 'CV':
        #    build_arg_list_cv(self)
        # elif self.Model_selection == 'IC':
        #    build_arg_list_ic(self)
        # else:
        #     raise ValueError("Model_selection must be either 'CV' or 'IC'")
        
        
        print(f"Starting parallel processing with {self.cfg.n_process} processes...")
        results= self.parallel_execution() 
        print("Parallel processing completed.")
        return results
    

    def parallel_execution(self):
    
            self.storage = {}

            pool = Pool(self.cfg.n_process)
            async_results = [
                pool.apply_async(self.worker, kwds={'node': self.nodes_list[i]})
                for i in range(len(self.nodes_list))
            ]
            pool.close()
            
            for i, async_result in enumerate(tqdm(async_results, desc="Processing nodes", unit="node")):
                try:
                    result = async_result.get(timeout=self.cfg.timeout_per_node) 
                except TimeoutError:
                    print(f"Timeout occurred for node {i}")
                    self.storage[i]=None
                    continue 
                
                if self.Model_selection=='CV':
                    cv_error, node = result
                    self.storage[node] = [cv_error]
                else:
                    print(f"finished node {i}")
                    bic, aic, node = result
                    self.storage[node] = [bic,aic]
            print("All nodes have been processed or timed out.")
            pool.terminate()
            pool.join()
                  
            return self.storage 

            
    def worker(self, node):
        if self.cfg.formulation == 'regional':
            from models.regional_model.information_criteria.run_experiment_ic import MainLoop as MainLoop
            model_loop = MainLoop(self, node)
            BIC, AIC, node= model_loop.run_experiment()
            return BIC, AIC, node
        # else:
        #     if self.Model_selection == 'CV':
        #         from models.global_model.cross_validation.run_experiment_cv import MainLoop as MainLoop
        #         model_loop = MainLoop(*args)
        #         cv_error, node, *unused = model_loop.run_experiment()
        #         return cv_error, node
        #     else:  
        #         from models.global_model.information_criteria.run_experiment_ic import MainLoop as MainLoop
        #         model_loop = MainLoop(*args)
        #         if args[-1] is not None:
        #             # Monte Carlo experiment
        #             Holdout_error, BIC, AIC, node, _, _ = model_loop.run_experiment()
        #         else:
        #             Holdout_error, BIC, AIC, node= model_loop.run_experiment()
        #         return Holdout_error, BIC, AIC, node

    
