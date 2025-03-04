def initialize_parameters(self):
      
        self.individuals = {}
        self.N = {}
        self.noObs = {}

        self.time_periods_na = {}
        self.time_periods_not_na = {}

        self.in_sample_pred = {}
        self.R2 = {}
        self.MSE = {}

        self.Min = {}
        self.Max = {}
        
        self.x_train_np = {}
        self.y_train_df = {}

        self.x_train_transf = {}
        self.y_train_transf = {}


        self.mask = {}

        self.losses = None
        self.epochs = None
        self.params = None
        self.BIC = None

        self.model_pred = None

        # Preparing data - getting the keys from the dictionaries, region names
        self.regions = list(self.x_train[0].keys())
        
        #number of regions
        self.no_regions = len(self.regions)

        #number of time periods, and the specific time periods
        self.T = self.x_train[0][self.regions[0]].shape[0]
        self.time_periods = self.x_train[0][self.regions[0]].index.values