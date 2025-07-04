from simulations.simulation_functions.bias import calculate_bias
import multiprocessing as mp

def setup_manager(self):
        self.manager=mp.Manager()
        self.bias_dict = self.manager.dict({k: 0 for k in self.cfg.mc.breakpoints})
        self.lock = self.manager.Lock()
    
