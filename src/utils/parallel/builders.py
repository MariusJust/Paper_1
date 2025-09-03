
 
def build_arg_list_cv(self):
        self.arg_list=[(
        self.nodes_list[i], 
        self.no_inits, 
        self.seed_value, 
        self.lr,
        self.min_delta, 
        self.patience, 
        self.verbose, 
        self.dropout,
        self.n_splits, 
        self.cv_approach, 
        self.penalty,
        self.n_countries, 
        self.time_periods,
        self.country_trends,
        self.dynamic_model,
        self.data
    ) for i in range(len(self.nodes_list))]


def build_arg_list_ic(self):
        self.arg_list=[(
        self.nodes_list[i],
        self.no_inits, 
        self.seed_value, 
        self.lr,
        self.min_delta, 
        self.patience, 
        self.verbose, 
        self.dropout,
        self.n_countries, 
        self.time_periods,
        self.penalty,
        self.country_trends,
        self.dynamic_model,
        self.data
        ) for i in range(len(self.nodes_list))]
            
  
def build_arg_list_mc(self):
    from simulations.simulation_functions import simulate
    if self.model == "NN":
        self.rep_args = [
            {
                "model": self.model,                               # keep for branching
                "node": self.nodes_list[self.node_index],
                "no_inits": self.cfg.instance.no_inits,
                "seed_value": self.cfg.instance.seed_value + rep + 1,  
                "lr": self.cfg.instance.lr,
                "min_delta": self.cfg.instance.min_delta,
                "patience": self.cfg.instance.patience,
                "verbose": self.cfg.instance.verbose,
                "dropout": self.cfg.instance.dropout,
                "n_splits": self.cfg.instance.n_splits,
                "cv_approach": self.cfg.instance.cv_approach,
                "penalty": self.cfg.instance.penalty,
                "n_countries": self.cfg.instance.n_countries,
                "time_periods": self.cfg.instance.time_periods,
                "country_trends": self.cfg.instance.country_trends,
                "model_selection":self.cfg.instance.model_selection,
                "dynamic_model":self.cfg.instance.dynamic_model,
                "data": simulate(
                    seed=self.cfg.instance.seed_value + rep + 1,
                    n_countries=self.cfg.instance.n_countries,
                    n_years=63,
                    specification=self.specification,
                    add_noise=True,
                    sample_data=True,
                    dynamic=self.cfg.instance.dynamic_model
                )
            }
            for rep in range(self.cfg.mc.reps)
        ]
    else:
        self.rep_args = [
        {
            "model": self.model,
            "data": simulate(
                seed=self.cfg.instance.seed_value + rep + 1,
                n_countries=self.cfg.instance.n_countries,
                n_years=63,
                specification=self.specification,
                add_noise=True,
                sample_data=True
            )
        }
        for rep in range(self.cfg.mc.reps)
    ]
    