import multiprocessing as mp
from models.global_model.information_criteria.run_experiment_ic import MainLoop
from .builders import build_arg_list_mc
from .manager import setup_manager


class MultiprocessingMC:
    """
    Class to handle multiprocessing for Monte Carlo simulations.
    """
    
    def __init__(self, node_index, nodes_list, cfg, specification):
        self.node_index = node_index
        self.nodes_list = nodes_list
        self.cfg = cfg
        self.specification = specification
        
    
    def run(self):
        
        #manager for shared data dictionary
        setup_manager(self)
        
        #builds arguments list to pass to the mc_worker
        build_arg_list_mc(self)
        
        self.counter= self.manager.Value('i', 0)

        #define the number of processes to use
        pool = mp.Pool(processes=self.cfg.instance.n_process)
        
        self.all_surfaces = []
        self.all_country_FE = []
        
        def callback_one(result):
            best_surface, country_FE = result
            with self.lock:
                self.counter.value += 1
                self.all_surfaces.append(best_surface)
                self.all_country_FE.append(country_FE)
                

        # apply_async expects a function and its arguments, so we need to pass the mc_worker function and the args as a tuple. I also added a callback function to handle breakpoints and bias calculation.
        for args_dict in self.rep_args:
            pool.apply_async(
                mc_worker,                 
                args=(args_dict, ),            # pass each dict as keyword‐args to mc_worker
                callback=callback_one      
            )

        pool.close()
        pool.join()

        return self.all_surfaces, self.all_country_FE

        
def mc_worker(args):
    from utils import create_pred_input
    pred_input, *unused=create_pred_input(True)
    main_loop= MainLoop(*args)
    *unused, best_surface, country_FE = main_loop.run_experiment()

    return best_surface.predict(pred_input).reshape(-1,), country_FE


        

