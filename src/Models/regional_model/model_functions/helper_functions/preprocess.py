import numpy as np
import pandas as pd



def Preprocess(self):
    import numpy as np
    import pandas as pd
     
    compute_x_train_fullSample(self, self.x_train)      
    compute_y_train_fullSample(self, self.y_train)
    compute_mask_fullSample(self, self.x_train)
    

################################### Full Sample Preprocessing Functions ###################################
def compute_x_train_fullSample(self, x_train):
    
     for key, var in x_train.items():
            for region in self.regions:
                x_df_var = self.x_train[key][region]
                x_np_var = x_df_var.values
                self.x_train_transf[key][region] = np.array(x_df_var.copy())
                
                # Compute statistics for each region
                region_stats = compute_stats(x_np_var)
                for key1, val in region_stats.items():
                        getattr(self, key1.capitalize())[key][region] = val
                        
                #compute the number of observations, only for the first variables as the input variables are the same
                if key == 'temp':
                    self.individuals[region] = x_df_var.columns.values
                    self.N[region] =len(self.individuals[region]) 
                      
                    #count number of time periods with at least one observation
                    self.time_periods_not_na[region] = np.sum(~np.isnan(x_np_var), axis=1) > 0
                    self.time_periods_na[region] = np.sum(~self.time_periods_not_na[region])
                
                    self.noObs[region] = self.N[region]*self.T - np.isnan(x_np_var).sum()


def compute_y_train_fullSample(self):
        for region in self.regions:
                y_df = self.y_train[region]
                y_np = y_df.values
                self.y_train_transf[region] = np.array(y_df.copy())
                self.y_train_df[region]=y_df.copy()
                
                # Compute statistics for each region
                region_stats = compute_stats(y_np)
                for key1, val in region_stats.items():
                    getattr(self, key1.capitalize())[region] = val
        
        
def compute_mask_fullSample(self, x_train):

     for region in self.regions:
            # Create a mask for each region
            self.mask[region] = np.isnan(x_train[region])


def compute_stats(arr):
        """
        Compute basic statistics on the array, ignoring NaNs.
        Returns a dict with keys: min, max, quant025, quant05, quant95, quant975.
        """
        return {
            'min': np.nanmin(arr),
            'max': np.nanmax(arr),
            'quant025': np.nanquantile(arr, 0.025),
            'quant05': np.nanquantile(arr, 0.05),
            'quant95': np.nanquantile(arr, 0.95),
            'quant975': np.nanquantile(arr, 0.975),
        }

