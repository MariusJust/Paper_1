from multiprocessing import Pool
from tqdm import tqdm
from multiprocessing import TimeoutError
from .builders import build_arg_list_cv, build_arg_list_ic
import ast


class Multiprocess:
    
    """
    This class is designed to run either cross-validation (CV) or information criteria (IC) based model selection in parallel.
    It initializes with configuration parameters (from the config folder) and data, builds the argument list for each node, and executes the training in parallel.
    The results are stored in a dictionary where keys are node indices and values are lists containing either cross-validation errors or BIC/AIC values.
    """
    def __init__(self, cfg, data=None):
        self.Model_selection = cfg.model_selection
        self.nodes_list = [ast.literal_eval(s) for s in cfg.nodes_list]
        self.no_inits = cfg.no_inits
        self.seed_value = cfg.seed_value
        self.lr = cfg.lr
        self.min_delta = cfg.min_delta
        self.patience = cfg.patience
        self.verbose = cfg.verbose
        self.dropout = cfg.dropout
        self.n_splits = cfg.n_splits
        self.n_process = cfg.n_process
        self.cv_approach = cfg.cv_approach
        self.penalty = cfg.penalty
        self.timeout_per_node = cfg.timeout_per_node
        self.n_countries = cfg.n_countries
        self.time_periods = cfg.time_periods
        self.data = data
        self.country_trends = cfg.country_trends
        self.dynamic_model = cfg.dynamic_model

    def run(self):
        if self.Model_selection == 'CV':
           build_arg_list_cv(self)
        elif self.Model_selection == 'IC':
           build_arg_list_ic(self)
        else:
            raise ValueError("Model_selection must be either 'CV' or 'IC'")
        
        results= self.parallel_execution() 
        return results
    

    def parallel_execution(self):
    
            self.storage = {}

            pool = Pool(self.n_process)
            async_results = [
                pool.apply_async(self.worker, args=(args,))
                for args in self.arg_list
            ]
            pool.close()
            
            for i, async_result in enumerate(tqdm(async_results, desc="Processing nodes", unit="node")):
                try:
                    result = async_result.get(timeout=self.timeout_per_node) 
                except TimeoutError:
                    print(f"Timeout occurred for node {i} with args {self.arg_list[i]}")
                    self.storage[i]=None
                    continue 
                
                if self.Model_selection=='CV':
                    cv_error, node = result
                    self.storage[node] = [cv_error]
                else:
                    bic,aic,node = result
                    self.storage[node] = [bic,aic]

            pool.terminate()
            pool.join()
            return self.storage 

            
    def worker(self, args):
        if self.Model_selection == 'CV':
            from models.global_model.cross_validation.run_experiment_cv import MainLoop as MainLoop
            model_loop = MainLoop(*args)
            cv_error, node, *unused = model_loop.run_experiment()
            return cv_error, node
        else:  
            from models.global_model.information_criteria.run_experiment_ic import MainLoop as MainLoop
            model_loop = MainLoop(*args)
            if args[-1] is not None:
                # Monte Carlo experiment
                BIC, AIC, node, _, _ = model_loop.run_experiment()
            else:
                BIC, AIC, node= model_loop.run_experiment()
            return BIC, AIC, node

 
