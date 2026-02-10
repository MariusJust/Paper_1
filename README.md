
## A Neural Network Approach to the Growth Relationships**

This repository contains the full source code, data pipeline, and replication material for the paper.

The project implements a flexible neural-network–based panel data model to study how temperature and precipitation jointly affect economic growth. The framework relaxes restrictive parametric assumptions common in the climate–growth literature while retaining country and time fixed effects and a rigorous model-selection strategy.


## Reproducing the environment (VS Code Devcontainer)

1. Install Docker desktop and VS Code
2. Install the “Dev Containers” extension in VS Code
3. Open this repository in VS Code and make sure you have a running engine in Docker
4. Press `F1` → **Dev Containers: Reopen in Container** 

Dependencies install automatically via `postCreateCommand`.

## Reproducing the environment (Docker)

1. Install docker on your machine 
2. run "docker build -t paper1 -f .devcontainer/Dockerfile ." from the command line 
3. run "docker run --rm -it -v "$(pwd)":/workspaces/Paper_1 -w /workspaces/Paper_1 paper1" from the command line. 

## Reproducing the results 

In the notebooks folder there exist 3 files for the global model, regional model and Monte Carlo simulation respectively. By running these notebooks, you can recreate the results shown in the paper. Note that the network weights are available in the results folder


## Contact

Marius Just
mjust@econ.au.dk
PhD Student, Econometrics
Aarhus University

For questions regarding the code, data construction, or replication, please open an issue or contact directly.


