## Overview
This project develops a **neural network–based panel data model** to estimate how **temperature**, **precipitation**, and other controls drive **log growth per capita** across countries worldwide. Traditional panel regressions capture fixed effects but struggle with complex nonlinearities; here we leverage the universal approximation capacity of ANN architectures within a fixed‑effects framework. By integrating global climate observations with economic indicators, our model uncovers nuanced climate–growth relationships beyond quadratic or spline specifications.

## Features
- **Flexible Panel ANN**: Implements a fixed‑effects neural network that handles unobserved heterogeneity and nonlinear interactions.  
- **Multi‑Metric Climate Inputs**: Uses country‑level temperature and precipitation distributions to capture both mean and variance effects.  
- **Reproducible Pipeline**: Full data ingestion, preprocessing, model training, evaluation, and result visualization in a single CLI.  
- **Configurable Experiments**: Hyperparameters and data paths managed via versioned YAML files, enabling easy experimentation.

## Data
- **Raw data** (in `data/raw/`):
  - **Temperature**: Monthly gridded averages from Berkeley Earth.  
  - **Precipitation**: Global Precipitation Climatology Centre (GPCC) monthly totals.  
  - **Economic**: World Bank log GDP per capita (PPP).  
- **Processed data** (in `data/processed/`): cleaned, aggregated to annual country‑level panels and merged for modeling.  
- **Note on precipitation**: Prior work often finds precipitation’s direct effect on growth statistically indistinct; our model re‑examines this within a nonlinear ANN setting.

## Modeling Approach
1. **Feature Engineering**  
   - Compute country‑year means and variances of temperature and precipitation.  
2. **Panel ANN Architecture**  
   - Shared base classes and registry in `src/models/base.py` & `registry.py`.  
   - Each model variant in its own subfolder (e.g. `fixed_effects_ann/`) with `architecture.py`, `train.py`, `predict.py`, and `config.yaml`.  
3. **Training & Evaluation**  
   - 5‑fold country cross‑validation to test out‑of‑sample performance against fixed‑effects OLS and splines.  
   - Metrics: RMSE, MAE, and elasticity estimates of climate variables.  
4. **Interpretation**  
   - Use SHAP values to decompose temperature vs. precipitation contributions to growth predictions.

## Results
- **Nonlinear Temperature Effects**  
  The ANN uncovers concave temperature–growth relationships with an inflection around 20 °C, consistent with prior findings.  
- **Precipitation Insights**  
  While linear models show null precipitation impacts, the ANN reveals threshold effects during extreme wet/dry years.  
- **Model Performance**  
  Achieves a ~15 % lower RMSE versus fixed‑effects OLS and quadratic spline benchmarks.

## Project Structure
