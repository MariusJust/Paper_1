
## A Neural Network Approach to the Growth Relationships**

This repository contains the full source code, data pipeline, and replication material for the paper.

The project implements a flexible neural-network–based panel data model to study how temperature and precipitation jointly affect economic growth. The framework relaxes restrictive parametric assumptions common in the climate–growth literature while retaining country and time fixed effects and a rigorous model-selection strategy.

---

<!-- ## Project Overview

Empirical studies of climate impacts on economic growth typically rely on low-order polynomial specifications with prespecified interaction terms. While tractable, these models impose strong global shape restrictions and are frequently used for forward-looking assessment and policy analysis, where extrapolation beyond the estimation sample is unavoidable.

This project develops a fully data-driven alternative using feedforward neural networks to approximate the unknown climate–growth relationship in a large panel of countries and years. The approach:

- allows for rich nonlinearities and interactions between temperature and precipitation,  
- retains country and time fixed effects,  
- uses an expanding-window cross-validation scheme aligned with out-of-sample predictive objectives, and  
- produces interpretable prediction surfaces comparable to existing parametric benchmarks.

The codebase is structured to support full replication of all empirical results, figures, and simulation exercises in the paper.

--- -->

## Repository Structure
Paper_1/
├── Benchmark/ # Parametric benchmark models and comparison routines
├── config/ # Configuration files (model specs, training settings)
├── data/
│ ├── raw/ # Raw input data (GDP, climate, population grids)
│ └── processed/ # Cleaned country-year panel datasets
├── notebooks/ # Exploratory analysis and diagnostic notebooks
├── results/ # Saved model outputs, figures, and tables
├── scripts/ # Executable scripts for training, evaluation, and plotting
├── src/ # Core source code (models, training loops, utilities)
├── .devcontainer/ # Reproducible development environment
├── requirements.txt # Python dependencies
└── README.md




##Contact

Marius Just
mjust@econ.au.dk
PhD Student, Econometrics
Aarhus University

For questions regarding the code, data construction, or replication, please open an issue or contact directly.


