# Data-driven model reduction of agent-based systems using the Koopman generator

This repository contains python codes for the articel "Data-driven model reduction of agent-based systems using the Koopman generator" by Jan-Hendrik Niemann, Stefan Klus and Christof Schütte.

https://arxiv.org/abs/2012.07718

## How to use?

There are three models predefined:

1. A voter model defined as a Markov jump process, `VoterModel.py`
2. An extened voter model defined on arbitrary networks, `ExtendedVoterModel.py`
3. A spatial predator-prey model, `PredatorPreyModel.py`

The use is as follows:

1. Create measurements with `demo_data_generation.py`. The script illustrates the procedure using the agent-based model in `VoterModel.py`. There are some pre-generated measurements in the directory `data/raw`.
2. Process the data to obtain *point-wise* estimates of drift and diffusion. Use gEDMD to learn a global description. The procedure is demonstrated in `demo_post_processing.py`. There are some post-processed measurements in the directory `data/processed`.
3. The reduced stochastic differential equation can be simulated. This is demonstrated in `demo_reduced_SDE.py`.

## Additional Requirements

The codes require the ***d3s - data-driven dynamical systems toolbox***: https://github.com/sklus/d3s