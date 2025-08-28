import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# ----------------------------
# Configuration / options
# ----------------------------
specification = "country_trends"  # choose either "country_trends" or "base"

# pandas display options (similar to R options)
pd.set_option('display.float_format', lambda x: f"{x:.5f}")

# ----------------------------
# Load data
# ----------------------------
dataset = pd.read_excel('../data/MainData.xlsx')

# ----------------------------
# Prepare variables
# ----------------------------
dataset['temp'] = dataset['TempPopWeight']
dataset['temp_sq'] = dataset['temp'] ** 2

# precipitation (scaled) and quadratic term
if 'PrecipPopWeight' in dataset.columns:
    dataset['precip'] = dataset['PrecipPopWeight'] / 1000.0
else:
    raise KeyError('PrecipPopWeight column not found in the dataset')

dataset['precip_sq'] = dataset['precip'] ** 2

# find _yi_ and _y2_ columns 
yi_vars = [c for c in dataset.columns if '_yi_' in c]
y2_vars = [c for c in dataset.columns if '_y2_' in c]
all_additional_vars = yi_vars + y2_vars

years = dataset['Year'].unique()


# ----------------------------
# Build formula
# ----------------------------
# Base formula with categorical ISO and Year (C() tells patsy/statsmodels to treat as categorical)
base_formula = 'GrowthWDI ~ temp + precip + temp_sq + precip_sq + precip*temp + precip_sq*temp + temp_sq*precip + temp_sq*precip_sq + C(ISO)'
base_formula = 'GrowthWDI ~ temp + precip + temp_sq + precip_sq + C(ISO) + C(Year)'

if specification == 'country_trends' and len(all_additional_vars) > 0:
    # add the additional time trend variables
    extra = ' + '.join(all_additional_vars)
    formula = base_formula + ' + ' + extra
elif specification == 'base':
    formula = base_formula
else:
    raise ValueError("Invalid specification. Choose either 'country_trends' or 'base'.")

print('\nUsing formula:')


# ----------------------------
# Fit the model
# ----------------------------

# fit OLS on the cleaned model_data
model_result = smf.ols(formula=formula, data=dataset).fit()

print('\nModel fit summary:')
print(model_result.summary())

model_result.params 

#write the country fixed effects, time fixed effects and country time trends to a csv 

#country fixed effects
country_FE=model_result.params.filter(like='C(ISO)').reset_index()
np.save('country_FE.npy', country_FE, allow_pickle=True)

#time fixed effects
time_FE=model_result.params.filter(like='C(Year)').reset_index()
np.save('time_FE.npy', time_FE, allow_pickle=True)

#linear time trend 
linear_time_trend=model_result.params.filter(like='X_yi_').reset_index()
np.save('linear_time_trend.npy', linear_time_trend, allow_pickle=True)

#quadratic time trend
quadratic_time_trend=model_result.params.filter(like='X_y2_').reset_index()
np.save('quadratic_time_trend.npy', quadratic_time_trend, allow_pickle=True)