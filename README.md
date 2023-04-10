# RL_planner
RL-based planner dev in f1tenth gym env.

Development is based on the [repository of the F1TENTH Gym environment](https://github.com/f1tenth/f1tenth_gym). 
You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Installation

For the installation and configuration of Anaconda, please check the readme file of [my LQR repo](https://github.com/derekhanbaliq/LQR-based-Path-Tracking). 

Create the environment as follows:
```bash
cd <repo_name>  # navigate to the root directory of this project
conda create -n f110_rl-planner python=3.8  # create a new conda environment with Python 3.8
conda activate f110_rl-planner  # activate the environment
pip install -e .  # install the dependencies for F1TENTH gym.
pip install -r requirements.txt  # install other dependencies
```

## Usage

WIP
