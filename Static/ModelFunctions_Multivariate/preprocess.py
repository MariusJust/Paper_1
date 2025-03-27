import numpy as np
import pandas as pd



def Preprocess(self, x_train):
        """
        Preprocess the training data, which is now provided as global data only.
        x_train is assumed to be a list of DataFrames.
        The first DataFrame (id 0) is used to initialize global aggregates,
        and any subsequent DataFrame is appended (only for x_train; y_train is set once).
        """

        self.N['global']=len(x_train[0]['global'].columns)
        # Initialize global x_train_transf with a copy of the first DataFrame.
        self.x_train_transf_temp['global'] = x_train[0]['global'].copy()
        self.x_train_transf_precip['global'] = x_train[1]['global'].copy()

        # Assume that self.y_train has been set already.
        self.y_train_transf['global'] = self.y_train['global'].copy()

        # Store the list of country codes (columns) as individuals.
        self.individuals['global'] = list(self.x_train_transf_temp['global'].columns)

        # Identify time periods with at least one non-NaN value.
        non_na = np.sum(~np.isnan(self.x_train_transf_temp['global']), axis=1) > 0
        self.time_periods_not_na['global'] = non_na
        self.time_periods_na['global'] = np.sum(~non_na)

        
        self.x_train_transf_temp['global'] = np.array(self.x_train_transf_temp['global'])
        self.x_train_transf_precip['global'] = np.array(self.x_train_transf_precip['global'])
        self.y_train_transf['global'] = np.array(self.y_train_transf['global'])
        self.y_train_df['global'] = np.array(self.y_train['global'])
        
        self.noObs['global'] = len(self.individuals['global']) * self.T - np.isnan(self.x_train_transf_temp['global']).sum()
        
         # Get the global minimum and maximum (ignoring NaNs).
        self.Min['global'] = np.nanmin(self.x_train_transf_temp['global'])
        self.Max['global'] = np.nanmax(self.x_train_transf_temp['global'])
        
        # Create a mask of missing data.
        self.mask['global'] = np.isnan(self.x_train_transf_temp['global'])
