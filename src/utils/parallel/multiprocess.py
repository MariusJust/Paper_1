from multiprocessing import Pool
from tqdm import tqdm
from multiprocessing import TimeoutError
from .builders import build_arg_list_cv, build_arg_list_ic


class Multiprocess:
    
    def __init__(self, Model_selection, nodes_list, no_inits,
                 seed_value, lr, min_delta, patience,
                 verbose, dropout, n_splits,
                 n_process, formulation, cv_approach, penalty, timeout_per_node,
                 n_countries, time_periods,
                 data=None):
        
        
        self.Model_selection = Model_selection
        self.nodes_list = nodes_list
        self.no_inits = no_inits
        self.seed_value = seed_value
        self.lr = lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.dropout = dropout
        self.n_splits = n_splits
        self.n_process = n_process
        self.formulation = formulation
        self.cv_approach = cv_approach
        self.penalty = penalty
        self.timeout_per_node = timeout_per_node
        self.n_countries = n_countries
        self.time_periods = time_periods
        self.data = data
        
        
    def run(self):
        if self.Model_selection == 'CV':
           build_arg_list_cv(self)
        elif self.Model_selection == 'IC':
           build_arg_list_ic(self)
        else:
            raise ValueError("Model_selection must be either 'CV' or 'IC'")
        
        results= self.parallel_execution(self.arg_list)
        
        print("finished calculating")
        return results
    

    def parallel_execution(self, arg_list):
    
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
                    print(f"Timeout occurred for node {i} with args {arg_list[i]}")
                    self.storage[i]=None
                    continue 
                
                if self.Model_selection=='CV':
                    cv_error, node = result
                    self.storage[node] = [cv_error]
                else:
                    bic,aic,node,_,_ = result
                    self.storage[node] = [bic,aic]

            pool.terminate()
            pool.join()

            return self.storage 

            
    def worker(self, args):
        if self.Model_selection == 'CV':
            from models.global_model.cross_validation.run_experiment_cv import MainLoop as MainLoop
            model_loop = MainLoop(*args)
            return model_loop.run_experiment()
        
        else:  
            from models.global_model.information_criteria.run_experiment_ic import MainLoop as MainLoop
            model_loop = MainLoop(*args)
            return model_loop.run_experiment()

 
