from collections import defaultdict
def initialize_parameters(self):
      
        self.individuals = {}
        self.N = {}
        self.noObs = {}

        self.time_periods_na = {}
        self.time_periods_not_na = {}

        self.in_sample_pred = {}
        self.R2 = {}
        self.MSE = {}
        
        self.alpha = defaultdict(dict)
        self.beta = defaultdict(dict)
        self.Min = defaultdict(dict)
        self.Max = defaultdict(dict)
        self.Quant025 = defaultdict(dict)
        self.Quant05 = defaultdict(dict)
        self.Quant95 = defaultdict(dict)
        self.Quant975 = defaultdict(dict)
  
        self.y_train_df = defaultdict(dict)

        self.x_train_transf = defaultdict(dict)
        
        self.y_train_transf = defaultdict(dict)
        self.mask = defaultdict(dict)

        self.losses = None
        self.epochs = None
        self.params = None
        self.BIC = None
        self.AIC = None
        self.country_FE= None
        

        self.model_pred = None

        # Preparing data - getting the keys from the dictionaries, region names
        self.regions = list(self.x_train['temp'].keys())
        
        #number of regions
        self.no_regions = len(self.regions)

        #number of time periods, and the specific time periods
        self.T = self.x_train['temp'][self.regions[0]].shape[0]
       
        self.time_periods = self.x_train['temp'][self.regions[0]].index.values
   