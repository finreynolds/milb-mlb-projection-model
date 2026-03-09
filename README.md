# MiLB to MLB Projection Model

This project trains a neural network using PyTorch to project basic MLB statistics from Minor League hitting metrics.

## Overview

The model:
- takes these MiLB stats as input: Age, GB/FB, LD%, HR/FB, SwStr%, BB%, K%
- predicts these MLB outcome probabilities: 1B, 2B, 3B, HR, BB, Out

The model is trained on historical player data by matching players across MiLB and MLB seasons. The dataset includes AA, AAA, and MLB leaderboards exported from FanGraphs spanning 2015 through 2025 (2020 AA and AAA seasons excluded). Modifying, moving, or deleting the datasets could break the program.

## Requirements

- Python 3.x
- PyTorch

Install dependencies:

```bash
pip install torch
```

## Running the model

Run:

```bash
python model.py
```

You will be prompted for:

- projection years
- starting MiLB level
- number of training simulations

After training, you can select a player from a given MiLB season and generate a projected MLB stat line.

## Example Output

```text
JIO MIER
AVG: 0.238
OBP: 0.279
SLG: 0.334
OPS: 0.613
HR: 11
```

## Data

Leaderboard CSV files should be placed in the `data/` folder.

MLB leaderboard CSV files should include the following columns:

Name, PA, 1B, 2B, 3B, HR, BB, PlayerId

Minor League leaderboard CSV files should include:

Name, Age, PA, GB/FB, LD%, HR/FB, SwStr%, BB%, K%, PlayerId
