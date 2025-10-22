import numpy as np
import pandas as pd



def Preprocess(self):
    import numpy as np
    import pandas as pd
     
    compute_x_train(self, self.x_train, train=True)      
    compute_y_train(self, self.y_train, train=True)
    compute_mask(self, self.x_train)
    
    if self.x_val is not None and self.y_val is not None:
        compute_x_train(self, self.x_val, train=False)      
        compute_y_train(self, self.y_val, train=False)
        
    
 
  
    
def compute_x_train(self, x_train, train=True):
     
        self.individuals['global'] = x_train[0]['global'].columns.values
        self.N['global'] = len(self.individuals['global'])   
                
        for key, var in enumerate(x_train):
            if train:
                self.x_train_transf[key]['global'] =np.array(x_train[key]['global'].copy())
                
                global_stats = compute_stats(self.x_train_transf[key]['global'])
                for key, val in global_stats.items():
                    getattr(self, key.capitalize())[key]['global'] = val
            else:
               self.x_val_transf[key]['global'] = np.array(x_train[key]['global'].copy())
               
               global_stats = compute_stats(self.x_val_transf[key]['global'])
               for key, val in global_stats.items():
                   getattr(self, key.capitalize())[key]['global'] = val

def compute_y_train(self, y_train, train=True):
    if train:
        self.y_train_df['global'] = y_train['global'].copy()
        self.y_train_transf['global'] = np.array(y_train['global'].copy())
        
        global_stats = compute_stats(self.y_train_transf['global'])
        for key, val in global_stats.items():
            getattr(self, key.capitalize())['global'] = val
    else:
        self.y_val_df['global'] = y_train['global'].copy()
        self.y_val_transf['global'] = np.array(y_train['global'].copy())
        global_stats = compute_stats(self.y_val_transf['global'])
        for key, val in global_stats.items():
            getattr(self, key.capitalize())['global'] = val


def compute_mask(self, x_train):

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

