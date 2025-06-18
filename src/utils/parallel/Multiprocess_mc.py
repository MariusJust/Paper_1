import multiprocessing as mp
from models.global_model.information_criteria.run_experiment_ic import MainLoop
from .builders import build_arg_list_mc
from .manager import setup_manager, callback_one



class MultiprocessingMC:
    """
    Class to handle multiprocessing for Monte Carlo simulations.
    """
    
    def __init__(self, node_index, nodes_list, no_inits, seed_value, lr, min_delta, patience, verbose, dropout, n_splits, n_processes, formulation, reps, specification, model_selection, penalty,  breakpoints, cfg):
        self.node_index = node_index
        self.nodes_list = nodes_list
        self.no_inits = no_inits
        self.seed_value = seed_value
        self.lr = lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.dropout = dropout
        self.n_splits = n_splits
        self.n_processes = n_processes
        self.formulation = formulation
        self.reps = reps
        self.penalty = penalty
        self.specification = specification # either Q_Leirvik, or quadratic
        self.model_selection = model_selection #either 'CV' or 'IC'
        self.breakpoints = breakpoints
        self.cfg = cfg
        
    
    def run(self):
        
        setup_manager(self)
        
        build_arg_list_mc(self)
        
        self.counter= self.manager.Value('i', 0)

        #define the number of processes to use
        pool = mp.Pool(processes=self.n_processes)
        
        self.all_weights = []

        # apply_async expects a function and its arguments, so we need to pass the mc_worker function and the args as a tuple. I also added a callback function to handle breakpoints and bias calculation.
        for args_dict in self.rep_args:
            pool.apply_async(
                mc_worker,                 
                args=(args_dict, ),            # pass each dict as keyword‐args to mc_worker
                callback=callback_one      
            )

        pool.close()
        pool.join()

        return self.all_weights, dict(self.bias_dict)

  
    
    
        
def mc_worker(self, args):
    
    main_loop= MainLoop(*args)
    *_, weights_BIC, _=main_loop.run_experiment()
    
    return weights_BIC


        

