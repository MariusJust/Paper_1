from simulations.simulation_functions import simulate
 
def build_arg_list_cv(self):
            self.arg_list=[(
        self.nodes_list[i], self.no_inits, self.seed_value, self.lr,
        self.min_delta, self.patience, self.verbose, self.dropout,
        self.n_splits, self.formulation, self.cv_approach, self.penalty,
        self.n_countries, self.time_periods
    ) for i in range(len(self.nodes_list))]


def build_arg_list_ic(self):
        self.arg_list=[(
        self.nodes_list[i], self.no_inits, self.seed_value, self.lr,
        self.min_delta, self.patience, self.verbose, self.dropout,
        self.formulation, self.n_countries, self.time_periods,
        self.penalty, self.data
        ) for i in range(len(self.nodes_list))]
            
  
def build_arg_list_mc(self):

    self.rep_args = [
        (

            self.nodes_list[self.node_index],  # node
            self.no_inits,
            self.seed_value + rep,  # replication_seed
            self.lr,
            self.min_delta,
            self.patience,
            self.verbose,
            self.dropout,
            self.model_selection,
            self.penalty,
            simulate(seed=self.seed_value + rep + 1, n_countries=196, n_years=63, specification=self.specification, add_noise=True)  # data
        )
        for rep in range(self.reps)
    ]
    