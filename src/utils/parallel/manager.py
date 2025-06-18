from simulations.simulation_functions.bias import calculate_bias
import multiprocessing as mp

def setup_manager(self):
        self.manager=mp.Manager()
        self.bias_dict = self.manager.dict({k: 0 for k in self.breakpoints})
        self.lock = self.manager.Lock()
    
def callback_one(self, result):
        with self.lock:
            self.counter.value += 1
            self.all_weights.append(result)
        
        if self.counter.value in self.breakpoints:
            best_node = self.nodes_list[self.node_index]
            self.bias_dict[self.counter.value] = calculate_bias(self.all_weights, self.specification, best_node, self.cfg)
        
        return result
    