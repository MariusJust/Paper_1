def setup_synthetic(model_selection, synthetic_data):
    """
    Modified version of setup() to obtain synthetic data.
    """
    growth, precip, temp = synthetic_data
    if model_selection == 'IC':
        return growth, precip, temp
    else:
        raise ValueError("Invalid model_selection argument. Use 'IC' or 'CV'.")

