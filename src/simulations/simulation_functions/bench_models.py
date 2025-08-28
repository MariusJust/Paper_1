def quadratic_model(data, P, T): 
    import statsmodels.formula.api as smf
    
    data['temperature_sq']=data['temperature']**2
    data['precipitation_sq']=data['precipitation']**2
    formula= 'delta_logGDP ~ temperature + precipitation + temperature_sq + precipitation_sq + C(CountryCode) + C(Year)'
    
    model = smf.ols(formula=formula, data=data).fit()
    prediction_surface = predict(model.params , P, T, "quadratic")
    
    return prediction_surface

def interaction_model(data, P, T):
    import statsmodels.formula.api as smf
    
    data['temperature_sq']=data['temperature']**2
    data['precipitation_sq']=data['precipitation']**2

    formula= 'delta_logGDP ~ temperature + precipitation + temperature_sq + precipitation_sq +temperature_sq*precipitation+temperature*precipitation_sq+precipitation*temperature+precipitation_sq*temperature_sq + C(CountryCode) + C(Year)'
    
    model = smf.ols(formula=formula, data=data).fit()
    prediction_surface = predict(model.params,  P, T, "interaction")

    return prediction_surface


def predict(coefficients, P, T, model):
    pred_temp = T
    pred_precip = P

    if model == "quadratic":
        return (
                coefficients['temperature'] * pred_temp +
                coefficients['precipitation'] * pred_precip +
                coefficients['temperature_sq'] * pred_temp**2 +
                coefficients['precipitation_sq'] * pred_precip**2 )
    elif model == "interaction":
        return (
                coefficients['temperature'] * pred_temp +
                coefficients['precipitation'] * pred_precip +
                coefficients['temperature_sq'] * pred_temp**2 +
                coefficients['precipitation_sq'] * pred_precip**2 +
                coefficients['temperature_sq:precipitation'] * pred_temp**2 * pred_precip +
                coefficients['temperature:precipitation_sq'] * pred_temp * pred_precip**2 +
                coefficients['precipitation:temperature'] * pred_precip*pred_temp +
                coefficients['precipitation_sq:temperature_sq'] * pred_precip**2 * pred_temp**2)