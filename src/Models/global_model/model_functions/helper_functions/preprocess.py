import numpy as np
import pandas as pd



def Preprocess(self):
    import numpy as np
    import pandas as pd
     
    compute_x_train(self, self.x_train)      
    compute_y_train(self, self.y_train)
    #assuming that missing values are the same in both input variables
    compute_mask(self, self.x_train)
    
 
  
    
def compute_x_train(self, x_train):
     
    if self.formulation == 'regional':
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
                    
    elif self.formulation == 'global':    
        self.individuals['global'] = x_train[0]['global'].columns.values
        self.N['global'] = len(self.individuals['global'])   
                
        for key, var in enumerate(x_train):
            self.x_train_transf[key]['global'] =np.array(x_train[key]['global'].copy())

            global_stats = compute_stats(self.x_train_transf[key]['global'])
            for key, val in global_stats.items():
                getattr(self, key.capitalize())[key]['global'] = val



def compute_y_train(self, y_train):

    if self.formulation == 'regional':
            for region in self.regions:
                y_df = self.y_train[region]
                y_np = y_df.values
                self.y_train_transf[region] = np.array(y_df.copy())
                self.y_train_df[region]=y_df.copy()
                
                # Compute statistics for each region
                region_stats = compute_stats(y_np)
                for key1, val in region_stats.items():
                    getattr(self, key1.capitalize())[region] = val
                    
    elif self.formulation == 'global':
            self.y_train_transf['global'] = y_train['global'].copy()
            self.y_train_transf['global'] = np.array(y_train['global'].copy())
            
    # Compute global statistics
            global_stats = compute_stats(self.y_train_transf['global'])
            for key, val in global_stats.items():
                getattr(self, key.capitalize())['global'] = val



def compute_mask(self, x_train):
    if self.formulation == 'regional':
        for region in self.regions:
            # Create a mask for each region
            self.mask[region] = np.isnan(x_train[region])
        
    elif self.formulation == 'global':
        self.mask['global'] = np.isnan(x_train[0]['global'])
        
        # also calculate the number of non-NaN values
        non_na = np.sum(~np.isnan(x_train[0]['global']), axis=1) > 0
        self.time_periods_not_na['global'] = non_na
        self.time_periods_na['global'] = np.sum(~non_na)
        self.noObs['global'] = self.N['global']* self.T - np.isnan(self.x_train_transf[0]['global']).sum()
        
        
        

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

