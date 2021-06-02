# Data-driven model reduction of agent-based systems using the Koopman generator

This repository contains python codes for the article "Data-driven model reduction of agent-based systems using the Koopman generator" by Jan-Hendrik Niemann, Stefan Klus and Christof Schütte.

Niemann J-H, Klus S, Schütte C (2021) Data-driven model reduction of agent-based systems using the Koopman generator. PLoS ONE 16(5): e0250970. https://doi.org/10.1371/journal.pone.0250970

## Agent-based models

There are three models predefined:

1. A voter model defined as a Markov jump process, `VoterModel.py`
2. An extended voter model defined on arbitrary networks, `ExtendedVoterModel.py`
3. A spatial predator-prey model, `PredatorPreyModel.py`

## How to use?

1. Create measurements with `demo_data_generation.py`. The script illustrates the procedure using the agent-based model in `VoterModel.py`. There are some pre-generated measurements in the directory `data/raw`.
2. Process the data to obtain *point-wise* estimates of drift and diffusion. Use gEDMD to learn a global description. The procedure is demonstrated in `demo_post_processing.py`. There are some post-processed measurements in the directory `data/processed`. Further data-sets are available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4522119.svg)](https://doi.org/10.5281/zenodo.4522119)
3. The reduced stochastic differential equation can now be simulated. This is demonstrated in `demo_reduced_SDE.py`.
4. The evaluation is demonstrated in `demo_evaluation.py`.

## Additional Requirements

The codes require the ***d3s - data-driven dynamical systems toolbox***: https://github.com/sklus/d3s