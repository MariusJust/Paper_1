import numpy as np
import pandas as pd


def Preprocess(self, x_train):
    
  # loop over both precipitation and temperature data
        for idy, var in enumerate(x_train): 
                print(f"idy: {idy}, var: {var}")
        
        
#if idy is 0, then we need to initialise the self.x_train dictionary, then when idy is 1, we need to append the data to the dictionary
                if idy == 0:
                                # loop over regions
                        for idx, region in enumerate(self.regions):
                                process_region(self, region, idy)
                                if idx == 0:
                                        initialize_global(self, region, idy)
                                else:
                                        update_global(self, region, idy)
                else:
                        #note: we do not want to update y_train again, we only want to update x_train
                        
                        for idx, region in enumerate(self.regions):
                                process_region(self, region, idy)
                                update_global_x(self, region, idy)
            
        finalize_global(self)

def process_region(self, region, index):
        """
        Process individual region data.
        """
        # Get the countries (individuals) in the region.
        self.individuals[region] = self.x_train[index][region].columns

        # Identify time periods with at least one non-NaN value.
        non_na = np.sum(~np.isnan(self.x_train[index][region]), axis=1) > 0
        self.time_periods_not_na[region] = non_na
        self.time_periods_na[region] = np.sum(~non_na)

        # Number of countries.
        self.N[region] = len(self.individuals[region])

        # Convert DataFrames to numpy arrays.
        self.x_train_np[region] = np.array(self.x_train[index][region])
        self.y_train_df[region] = np.array(self.y_train[region])

        # Total number of observations.
        self.noObs[region] = self.N[region] * self.T - np.isnan(self.x_train_np[region]).sum()

        # Get the minimum and maximum values.
        self.Min[region] = np.nanmin(self.x_train_np[region])
        self.Max[region] = np.nanmax(self.x_train_np[region])

        # Copy original data for transformations.
        self.x_train_transf[region] = np.array(self.x_train[index][region].copy())
        self.y_train_transf[region] = np.array(self.y_train[region].copy())

        # Create a mask for missing data.
        self.mask[region] = np.isnan(self.x_train_transf[region])

def initialize_global(self, region, index):
        """
        Initialize the global aggregates with the data from the first region.
        """
        self.individuals['global'] = list(self.individuals[region])
        self.time_periods_not_na['global'] = np.sum(~np.isnan(self.x_train[index][region]), axis=1) > 0
        self.x_train_transf['global'] = self.x_train[index][region].copy()
        self.y_train_transf['global'] = self.y_train[region].copy()

def update_global(self, region, index):
        """
        Update global aggregates by adding data from an additional region.
        """
        self.individuals['global'] += list(self.individuals[region])
        self.time_periods_not_na['global'] |= np.sum(~np.isnan(self.x_train[index][region]), axis=1) > 0
        self.x_train_transf['global'] = pd.concat([self.x_train_transf['global'], self.x_train[index][region]], axis=1)
        self.y_train_transf['global'] = pd.concat([self.y_train_transf['global'], self.y_train[region]], axis=1)
        
def update_global_x(self, region, index):
        """
        Update global aggregates by adding data from an additional region.
        """
        self.individuals['global'] += list(self.individuals[region])
        self.time_periods_not_na['global'] |= np.sum(~np.isnan(self.x_train[index][region]), axis=1) > 0
        self.x_train_transf['global'] = pd.concat([self.x_train_transf['global'], self.x_train[index][region]], axis=1)
        
        
        


def finalize_global(self):
        """
        Perform the final calculations for the global data.
        """
        # Compute global missing data periods.
        self.time_periods_na['global'] = np.sum(~self.time_periods_not_na['global'])
        
        # Sum the number of countries across regions.
        self.N['global'] = np.sum(list(self.N.values()))
        
        # Convert the global transformed data to numpy arrays.
        self.x_train_np['global'] = np.array(self.x_train_transf['global'])
        self.noObs['global'] = self.N['global'] * self.T - np.isnan(self.x_train_np['global']).sum()
        self.Min['global'] = np.nanmin(self.x_train_np['global'])
        self.Max['global'] = np.nanmax(self.x_train_np['global'])
        self.x_train_transf['global'] = np.array(self.x_train_transf['global'])
        self.y_train_transf['global'] = np.array(self.y_train_transf['global'])
        
        self.mask['global'] = np.isnan(self.x_train_transf['global'])
