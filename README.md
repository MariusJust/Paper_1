## Overview
This project develops a **neural network–based panel data model** to estimate how **temperature**, **precipitation**, and other controls drive **log growth per capita** across countries worldwide. Traditional panel regressions capture fixed effects but struggle with complex nonlinearities; here we leverage the universal approximation capacity of Artificial Neural Network architectures within a fixed‑effects framework. By integrating global climate observations with economic indicators, our model uncovers nuanced climate–growth relationships.

## Features
- **Flexible Panel Neureal Network*: Implements a fixed‑effects neural network that handles unobserved heterogeneity and nonlinear interactions.  
- **Multi‑Metric Climate Inputs**: Uses country‑level temperature and precipitation distributions to capture both mean and variance effects.  
- **Reproducible Pipeline**: Full data ingestion, preprocessing, model trainin an´d, evaluation
- **Configurable Experiments**: Hyperparameters and data paths managed via versioned YAML files, enabling easy experimentation.

## Data
- **Raw data** (in `data/raw/`):
  - **Temperature**: Monthly gridded averages from Berkeley Earth.  
  - **Precipitation**: Global Precipitation Climatology Centre (GPCC) monthly totals.  
  - **Economic**: World Bank log GDP per capita (PPP).  
- **Processed data** (in `data/processed/`): cleaned, aggregated to annual country‑level panels and merged for modeling.  
- **Note on precipitation**: Prior work often finds precipitation’s direct effect on growth statistically indistinct; our model re‑examines this within a nonlinear NN setting.

