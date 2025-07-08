import multiprocessing as mp
from .builders import build_arg_list_mc
from .manager import setup_manager
import time


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
        
        # record start time for runtime tracking
        self.start_time = time.time()

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
            best_surface, country_FE, runtime = result
            with self.lock:
                self.counter.value += 1
                self.all_surfaces.append(best_surface)
                self.all_country_FE.append(country_FE)
                
                #runtime tracking 
                runtime =runtime/60
                elapsed_time = (time.time() - self.start_time)/60
                print(f"Process {self.counter.value}/{self.cfg.mc.reps} completed in {runtime: .2f} min. Total elapsed time: {elapsed_time: .2f} min.")
                

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
    import time
    t0 = time.time()
    from utils import create_pred_input
    pred_input, *unused=create_pred_input(True)
    
    if len(args) == 14: #cv case
        from models.global_model.cross_validation.run_experiment_cv import MainLoop
        main_loop = MainLoop(*args)
        *unused, best_surface, country_FE= main_loop.run_experiment()
    else:
        from models.global_model.information_criteria.run_experiment_ic import MainLoop
        main_loop= MainLoop(*args)
        *unused, best_surface, country_FE = main_loop.run_experiment()
    
    runtime= time.time() - t0
  
    return best_surface.predict({"X_in": pred_input}, verbose=0).reshape(-1,), country_FE, runtime


        

