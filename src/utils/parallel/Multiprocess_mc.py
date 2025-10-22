import multiprocessing as mp
from .builders import build_arg_list_mc
from .manager import setup_manager
import time


class MultiprocessingMC:
    """
    Class to handle multiprocessing for Monte Carlo simulations.
    """
    
    def __init__(self, cfg, specification, model, node_index=None, nodes_list=None):
        self.node_index = node_index
        self.nodes_list = nodes_list
        self.cfg = cfg 
        self.specification = specification
        self.model = model
    
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
        
        def error_cb(exc):
            import traceback, sys
            print("WORKER ERROR (in child):", exc, file=sys.stderr)
            traceback.print_exc()

        def callback_one(result):
            try:
                best_surface, country_FE, runtime = result
                # ensure values exist
                try:
                    with self.lock:
                        self.counter.value += 1
                        self.all_surfaces.append(best_surface)
                        self.all_country_FE.append(country_FE)
                except AttributeError:
                    # fallback if lock is not a context manager
                    self.lock.acquire()
                    try:
                        self.counter.value += 1
                        self.all_surfaces.append(best_surface)
                        self.all_country_FE.append(country_FE)
                    finally:
                        self.lock.release()

                runtime = runtime/60
                elapsed_time = (time.time() - self.start_time)/60
                print(f"Process {self.counter.value}/{self.cfg.mc.reps} completed in {runtime: .2f} min. Total elapsed time: {elapsed_time: .2f} min.", flush=True)

            except Exception as e:
                import traceback
                print("Error in callback_one:", e)
                traceback.print_exc()

        # then in loop:
        for args_dict in self.rep_args:
            pool.apply_async(
                mc_worker,
                kwds=args_dict,
                callback=callback_one,
                error_callback=error_cb
            )


        pool.close()
        pool.join()

        return self.all_surfaces, self.all_country_FE

        
def mc_worker(**payload):

    import time
    t0 = time.time()
    from utils import create_pred_input
    import numpy as np
 
    
    data = payload.get("data")
    mean_T = np.mean(data["temperature"]); std_T = np.std(data["temperature"])
    mean_P = np.mean(data["precipitation"]); std_P = np.std(data["precipitation"])
    try:
        time_periods = payload.get("time_periods")
    except:
        time_periods=data['Year'].max().year-data['Year'].min().year +1
   
    
    if payload.get("dynamic_model"):
        pred_input, T, P = create_pred_input(mc=True, mean_T=mean_T, std_T=std_T, mean_P=mean_P, std_P=std_P, time_periods=time_periods)
    else:
        pred_input, T, P = create_pred_input(mc=True, mean_T=mean_T, std_T=std_T, mean_P=mean_P, std_P=std_P, time_periods=None)

    model = payload.get("model")
    model_selection=payload.get("model_selection")

    if model == "NN":
        
        if model_selection=='CV':
            from models.global_model.cross_validation.run_experiment_cv import MainLoop
            
            allowed = ("node","no_inits","seed_value","lr","min_delta","patience",
                    "verbose","dropout","n_splits","cv_approach","penalty",
                    "n_countries","time_periods","country_trends", "dynamic_model","data")
            mainloop_kwargs = {k: payload[k] for k in allowed if k in payload}

            main_loop = MainLoop(**mainloop_kwargs)
            *unused, best_surface, country_FE = main_loop.run_experiment()
            
        elif model_selection=='IC':
            from models.global_model.information_criteria.run_experiment_ic import MainLoop
            
            allowed = ("node","no_inits","seed_value","lr","min_delta","patience",
                    "verbose","dropout","penalty",
                    "n_countries","time_periods","country_trends", "dynamic_model","data")
            
            mainloop_kwargs = {k: payload[k] for k in allowed if k in payload}

            main_loop = MainLoop(**mainloop_kwargs)
            *unused, best_surface, country_FE = main_loop.run_experiment()
            
        
        
        runtime = time.time() - t0
        preds = best_surface.predict({"X_in": pred_input}, verbose=0).reshape(-1,)
        return preds, country_FE, runtime

    elif model == "Quadratic":
        from simulations.simulation_functions import quadratic_model
        predictions = quadratic_model(data, P, T)
        return predictions, None, time.time() - t0

    else:
        from simulations.simulation_functions import interaction_model
        predictions = interaction_model(data, P, T)
        return predictions, None, time.time() - t0
        

